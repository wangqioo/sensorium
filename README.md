# Sensorium

> *From raw sensor signals to emergent understanding — the complete sensory apparatus for embodied AI.*

Sensorium is a research architecture and implementation framework for building robots that perceive the world through **atomic sensory tokens** — discrete, self-supervised representations learned directly from raw sensor streams — which are then fed as a first-class vocabulary into a large language model.

The core idea: don't teach the LLM to understand "eye movement" or "stroking". Train the sensors to discover their own vocabulary. Then let the LLM learn what that vocabulary means.

---

## The Two-Stage Philosophy

```
Stage 1: Raw sensors → Atomic tokens      (data-driven, unsupervised)
Stage 2: Atomic tokens + text → LLM       (semantic emergence)
```

A dog turns its head at a sound before it "understands" the sound. Stage 1 is that reflex. Stage 2 is the thinking that follows.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        HARDWARE LAYER                        │
│   Camera │ Mic Array │ IMU (6-DOF) │ FSR Pressure Array    │
└────┬──────────┬───────────┬──────────────┬───────────────────┘
     │          │           │              │
┌────▼──────────▼───────────▼──────────────▼───────────────────┐
│              LAYER 1: ATOMIC ENCODER LAYER                    │
│  (Jetson Orin, TensorRT INT8, <10ms per modality)            │
│                                                               │
│  DINOv2-small       WavTokenizer    VQ-VAE-IMU  VQ-VAE-TAC  │
│  + VQ-GAN head      (pretrained)    (self-train) (self-train)│
│  10 VIS tokens/s    10 AUD tokens/s  2 IMU/s     5 TAC/s    │
└────┬──────────┬───────────┬──────────────┬───────────────────┘
     └──────────┴───────────┴──────────────┘
                            │  atomic token stream
              ┌─────────────▼─────────────┐
              │   LAYER 2: REFLEX ENGINE   │
              │   rule engine, <50ms       │
              │   atom pattern → action    │
              └─────────────┬─────────────┘
                            │  filtered token stream
              ┌─────────────▼─────────────┐
              │   LAYER 3: LLM REASONING  │
              │   Qwen2.5-7B (INT4)       │
              │   30s rolling context      │
              │   ~300ms inference cycle   │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   LAYER 4: ACTION OUTPUT  │
              │   voice / motion / state   │
              └───────────────────────────┘
```

---

## Four Sensor Modalities

### Vision
- **Encoder**: DINOv2-small (frozen) + VQ-GAN quantizer head
- **Output**: 10 `VIS` tokens/sec, codebook size 1024
- **Training**: VQ-GAN head fine-tuned on collected data, DINOv2 frozen
- **Edge**: TensorRT INT8, ~8ms/frame on Jetson Orin

### Audio
- **Encoder**: [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) (ICLR 2025)
- **Output**: 10 `AUD` tokens/sec (downsampled from 40 tokens/sec)
- **Training**: Use pretrained weights directly — no training needed
- **Edge**: TensorRT FP16, streaming with 32ms latency

### IMU (6-DOF Motion)
- **Encoder**: 1D CNN + VQ quantizer, initialized from [ImageBind](https://github.com/facebookresearch/ImageBind) IMU encoder
- **Output**: 2 `IMU` tokens/sec, codebook size 256
- **Training**: VQ-VAE self-supervised on 0.5s windows, aligned with ImageBind embeddings
- **Architecture reference**: [IMU-Video-MAE](https://github.com/mf-zhang/IMU-Video-MAE) (ECCV 2024)

### Tactile (FSR Pressure Array)
- **Hardware**: ~100 FSR sensors distributed over robot body, mapped to 16×8 virtual body grid
- **Encoder**: Spatial Conv2D + Temporal Conv1D + VQ quantizer
- **Output**: 5 `TAC` tokens/sec, codebook size 256
- **Training**: Spatial-temporal VQ-VAE, trained from scratch with cross-modal alignment loss
- **Key dimensions captured**: location (head/back/belly/side), pressure magnitude, coverage area, stroke velocity/direction, contact duration

---

## Token Vocabulary Design

```
Original LLM vocabulary:  ~151,936 tokens  (Qwen2.5-7B)

Added sensory tokens:
  [VIS_0000] ~ [VIS_1023]   → 1024 visual atom tokens
  [AUD_0000] ~ [AUD_1023]   → 1024 audio atom tokens
  [IMU_000]  ~ [IMU_255]    →  256 motion atom tokens
  [TAC_000]  ~ [TAC_255]    →  256 tactile atom tokens
  Special:   [SENSOR_START] [VIS_START] [AUD_START]
             [IMU_START] [TAC_START] [SENSOR_END]

Total: ~154,566 tokens
```

The LLM sees interleaved streams of sensory and text tokens:

```
[SENSOR_START]
[VIS_START][VIS_437][VIS_201][VIS_333]
[AUD_START][AUD_089][AUD_156]
[IMU_START][IMU_012]
[TAC_START][TAC_087][TAC_091]
[SENSOR_END]
[TXT] "今天好累啊" [/TXT]
...
```

It never learns what `TAC_087` "is." It learns what it *predicts*.

---

## Three-Stage Training Pipeline

### Stage 1 — Atomic Encoder Training (unsupervised)

Each modality encoder trained independently on raw sensor data. No labels.

| Modality | Loss functions | Duration |
|---|---|---|
| Vision | Reconstruction + temporal prediction | ~3 days GPU |
| Audio | Pretrained (WavTokenizer) | 0 |
| IMU | Reconstruction + prediction + ImageBind alignment | ~1 day |
| Tactile | Reconstruction + spatial continuity + temporal prediction | ~2 days |

**Anti-codebook-collapse**: EMA updates + commitment loss + periodic codebook reset for low-usage entries.

### Stage 2 — Cross-modal Alignment (ImageBind as anchor)

Use [ImageBind](https://github.com/facebookresearch/ImageBind) as a free semantic anchor. ImageBind already aligns vision, audio, and IMU into a shared space. We align our atom embeddings to it:

```
L_align = MSE(our_vis_atom_emb, ImageBind.encode_image(frame))
        + MSE(our_aud_atom_emb, ImageBind.encode_audio(clip))
        + MSE(our_imu_atom_emb, ImageBind.encode_imu(window))
```

Tactile uses cross-modal contrastive loss (co-occurring TAC and VIS/AUD atoms should be close).

### Stage 3 — LLM Multimodal Fine-tuning

Two sub-stages following [LLaVA](https://github.com/haotian-liu/LLaVA) protocol, tooled with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):

**Stage 3A** — Embedding alignment (LLM frozen, only new token embeddings trained)  
**Stage 3B** — LoRA full fine-tuning (rank=64, alpha=128) on multimodal interleaved sequences

---

## Reflex Engine

Fast path. No LLM. Pure rule lookup on the atom stream.

```python
reflex_rules = {
    "AUD_SUDDEN_LOUD":    (turn_head_to_sound,   priority=10),
    "TAC_HEAD_STROKE":    (enter_calm_mode,       priority=5),
    "TAC_SUDDEN_IMPACT":  (flinch_and_alert,      priority=9),
    "TAC_PAT_RHYTHM":     (wag_response,          priority=4),
    "IMU_FALL_DETECT":    (emergency_stop,         priority=10),
}
```

Atom IDs are filled in after Stage 1 training by inspecting the codebook. The reflex layer fires in <50ms — comparable to animal reflex arcs.

---

## Edge Deployment (Jetson Orin)

```
PyTorch → ONNX → TensorRT INT8 engine → ONNX Runtime (TensorRT backend)
```

| Model | Precision | Latency (Jetson Orin) |
|---|---|---|
| DINOv2-small + VQ-GAN | INT8 | ~10ms/frame |
| WavTokenizer | FP16 | ~5ms/chunk |
| VQ-VAE-IMU | INT8 | ~1ms/window |
| VQ-VAE-TAC | INT8 | ~2ms/window |
| Reflex Engine | — | <1ms |
| Qwen2.5-7B (INT4, llama.cpp) | INT4 | ~300ms/call |

Layer 1 + 2 complete in <20ms. Layer 3 runs asynchronously on a 300ms cycle.

---

## Implementation Roadmap

| Phase | Duration | Deliverable |
|---|---|---|
| **0** Data infrastructure | 2 weeks | ROS2 sensor nodes, FSR array bridge, synchronized recording |
| **1** Atomic encoder training | 3 weeks | VQ models for all 4 modalities, codebook quality validated |
| **2** Cross-modal alignment | 2 weeks | ImageBind-anchored atom embeddings, tactile contrastive alignment |
| **3** LLM fine-tuning | 4 weeks | Qwen2.5-7B with sensory token vocabulary, scene understanding eval |
| **4** Edge deployment | 3 weeks | TensorRT INT8, llama.cpp INT4, end-to-end latency validated |

---

## Key Open Source Dependencies

| Component | Repository | Notes |
|---|---|---|
| Visual backbone | [DINOv2](https://github.com/facebookresearch/dinov2) | Frozen encoder |
| Visual quantizer | [VQGAN-pytorch](https://github.com/dome272/VQGAN-pytorch) | Head only trained |
| Audio tokenizer | [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) | Used as-is |
| Cross-modal anchor | [ImageBind](https://github.com/facebookresearch/ImageBind) | Alignment signal |
| Tactile reference | [Sparsh](https://github.com/facebookresearch/sparsh) | Architecture reference |
| IMU reference | [IMU-Video-MAE](https://github.com/mf-zhang/IMU-Video-MAE) | Architecture reference |
| LLM fine-tuning | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | LoRA training |
| LLM inference | [llama.cpp](https://github.com/ggerganov/llama.cpp) | INT4 edge inference |
| Robot middleware | ROS 2 + Isaac ROS | Sensor integration |

---

## Repository Structure (Planned)

```
sensorium/
├── docs/                    # Architecture docs and papers
├── hardware/                # FSR array schematics, CAD, wiring guides
│   └── tactile_array/
├── data/                    # Data collection and preprocessing
│   ├── collector/           # ROS2 sensor nodes
│   └── preprocessor/        # Sync, format conversion
├── encoders/                # Stage 1: atomic encoder models
│   ├── visual/              # DINOv2 + VQ-GAN
│   ├── audio/               # WavTokenizer wrapper
│   ├── imu/                 # VQ-VAE for 6-DOF IMU
│   └── tactile/             # Spatial-temporal VQ-VAE
├── alignment/               # Stage 2: cross-modal alignment
│   └── imagebind_anchor/
├── llm/                     # Stage 3: multimodal LLM
│   ├── tokenizer_extension/ # Vocabulary expansion
│   ├── training/            # LLaMA-Factory configs
│   └── inference/           # llama.cpp integration
├── reflex/                  # Layer 2: reflex engine
└── deploy/                  # Edge deployment (TensorRT, ONNX)
    └── jetson/
```

---

## The Naming

*Sensorium* (Latin) — in neuroscience, the totality of an organism's sensory experience: all sensory inputs, the brain structures that process them, and the unified perceptual world they construct.

That's exactly what this is.

---

## Status

Early research / architecture phase. Contributions and discussion welcome.
