# Sensorium

> *从原始传感器信号到涌现式理解——具身 AI 的完整感知系统。*

Sensorium 是一套研究架构与实现框架，目标是让机器人通过**原子感知 Token** 来理解世界。所谓原子感知 Token，是直接从原始传感器数据流中以自监督方式学习出来的离散表征，然后作为第一等词汇喂给大语言模型。

核心理念：不去教 LLM 什么叫"眼神飘移"或"被抚摸"。先让传感器自己发现词汇，再让 LLM 去学这些词汇的含义。

---

## 两阶段哲学

```
第一阶段：原始传感器 → 原子 Token      （数据驱动，无监督）
第二阶段：原子 Token + 文字 → LLM      （语义涌现）
```

小狗听到声音会先转头，再去"理解"那是什么声音。第一阶段就是那个转头的反射；第二阶段才是随后的思考。

---

## 系统架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                          硬件层                              │
│      摄像头 │ 麦克风阵列 │ IMU（六轴）│ FSR 压力传感器阵列   │
└────┬──────────┬───────────┬──────────────┬───────────────────┘
     │          │           │              │
┌────▼──────────▼───────────▼──────────────▼───────────────────┐
│                 第一层：原子编码层                             │
│         （Jetson Orin，TensorRT INT8，每模态 <10ms）          │
│                                                               │
│  DINOv2-small      WavTokenizer    VQ-VAE-IMU  VQ-VAE-TAC   │
│  + VQ-GAN 量化头   （直接使用）    （自训练）   （自训练）    │
│  10 VIS token/秒   10 AUD token/秒  2 IMU/秒    5 TAC/秒     │
└────┬──────────┬───────────┬──────────────┬───────────────────┘
     └──────────┴───────────┴──────────────┘
                            │  原子 token 流
              ┌─────────────▼─────────────┐
              │       第二层：反射引擎      │
              │   规则引擎，延迟 <50ms     │
              │   原子模式 → 立即动作      │
              └─────────────┬─────────────┘
                            │  过滤后的 token 流
              ┌─────────────▼─────────────┐
              │      第三层：LLM 推理      │
              │   Qwen2.5-7B（INT4 量化）  │
              │   30 秒滚动上下文窗口       │
              │   推理周期 ~300ms          │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │       第四层：动作输出      │
              │   语音 / 运动 / 表情 / 状态 │
              └───────────────────────────┘
```

---

## 四个感知模态

### 视觉
- **编码器**：DINOv2-small（冻结）+ VQ-GAN 量化头
- **输出**：10 个 `VIS` token/秒，码本大小 1024
- **训练**：只训练 VQ-GAN 头部，DINOv2 权重冻结不动
- **边缘部署**：TensorRT INT8，Jetson Orin 上约 8ms/帧

### 听觉
- **编码器**：[WavTokenizer](https://github.com/jishengpeng/WavTokenizer)（ICLR 2025）
- **输出**：10 个 `AUD` token/秒（从 40 token/秒下采样）
- **训练**：直接使用预训练权重，无需训练
- **边缘部署**：TensorRT FP16，流式处理延迟 32ms

### IMU（六轴运动）
- **编码器**：1D CNN + VQ 量化器，用 [ImageBind](https://github.com/facebookresearch/ImageBind) 的 IMU 编码器初始化
- **输出**：2 个 `IMU` token/秒，码本大小 256
- **训练**：在 0.5 秒滑动窗口上做 VQ-VAE 自监督，损失函数与 ImageBind 嵌入对齐
- **架构参考**：[IMU-Video-MAE](https://github.com/mf-zhang/IMU-Video-MAE)（ECCV 2024）

### 触觉（FSR 压力传感器阵列）
- **硬件**：约 100 个 FSR 传感器分布于机器人身体，映射到 16×8 虚拟身体地图
- **编码器**：空间 Conv2D + 时间 Conv1D + VQ 量化器
- **输出**：5 个 `TAC` token/秒，码本大小 256
- **训练**：时空 VQ-VAE 从零自训练，加跨模态对齐损失
- **捕捉维度**：接触位置（头/背/腹/侧面）、压力大小、接触面积、抚摸速度/方向、持续时长

传感器分区示意：

```
     ┌──────────────────┐
     │   头部（4×2）     │  →  8 个感应点
     ├──────────────────┤
     │   背部（8×4）     │  →  32 个感应点
     ├──────────────────┤
     │   腹部（4×4）     │  →  16 个感应点
     ├────────┬─────────┤
     │ 左侧   │  右侧   │  →  各 8 个感应点
     │（2×4） │ （2×4） │
     └────────┴─────────┘
```

---

## Token 词汇表设计

```
原始 LLM 词汇量：  ~151,936 个 token（Qwen2.5-7B）

新增感官 token：
  [VIS_0000] ~ [VIS_1023]   →  1024 个视觉原子 token
  [AUD_0000] ~ [AUD_1023]   →  1024 个听觉原子 token
  [IMU_000]  ~ [IMU_255]    →   256 个运动原子 token
  [TAC_000]  ~ [TAC_255]    →   256 个触觉原子 token
  特殊 token：[SENSOR_START] [VIS_START] [AUD_START]
              [IMU_START] [TAC_START] [SENSOR_END]

总词汇量：~154,566 个
新 token 占比：~1.7%，对原模型影响极小
```

LLM 看到的输入是感官 token 与文字 token 的交织流：

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

LLM 永远不需要被告知 `TAC_087` 是什么。它只需要学会 `TAC_087` 能*预测*什么。

---

## 三阶段训练流程

### 第一阶段——原子编码器训练（无监督）

各模态编码器独立训练，只吃原始传感器数据，不需要任何标注。

| 模态 | 损失函数 | 预估训练时长 |
|---|---|---|
| 视觉 | 重建损失 + 时序预测损失 | ~3 天（GPU） |
| 听觉 | 直接使用预训练权重 | 0 |
| IMU | 重建 + 预测 + ImageBind 对齐 | ~1 天 |
| 触觉 | 重建 + 空间连续性 + 时序预测 | ~2 天 |

**防止码本崩塌**：EMA 更新 + commitment loss + 低频条目周期性重置。

### 第二阶段——跨模态对齐（以 ImageBind 为锚点）

用 [ImageBind](https://github.com/facebookresearch/ImageBind) 作为免费的语义锚——它已经把视觉、听觉、IMU 对齐到同一语义空间。我们的原子嵌入向它看齐：

```python
L_align = MSE(我们的视觉原子嵌入, ImageBind.encode_image(对应帧))
        + MSE(我们的听觉原子嵌入, ImageBind.encode_audio(对应音频))
        + MSE(我们的IMU原子嵌入,  ImageBind.encode_imu(对应窗口))
```

触觉没有 ImageBind 支持，改用跨模态对比损失：同一事件的 TAC 原子与对应的 VIS/AUD 原子在嵌入空间里拉近。

### 第三阶段——LLM 多模态微调

参考 [LLaVA](https://github.com/haotian-liu/LLaVA) 的两步协议，工具链使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)：

**第 3A 步**：嵌入对齐（冻结 LLM 主体，只训练新 token 的 embedding 层）

**第 3B 步**：LoRA 全量微调（rank=64，alpha=128），在多模态交织序列上训练

---

## 反射引擎

快速通路，完全不过 LLM，纯原子 token 规则查表。

```python
reflex_rules = {
    "AUD_SUDDEN_LOUD":    (转头朝向声源,     priority=10),
    "TAC_HEAD_STROKE":    (进入平静模式,      priority=5),
    "TAC_SUDDEN_IMPACT":  (缩避并发出警报,    priority=9),
    "TAC_PAT_RHYTHM":     (同频摆动响应,      priority=4),
    "IMU_FALL_DETECT":    (紧急停止,          priority=10),
}
```

各规则里的原子 ID，是第一阶段训练完成后，人工检查码本、找到对应模式的条目编号后填入的。只做一次，之后规则引擎的延迟 <50ms——与动物的脊髓反射弧相当。

---

## 边缘部署（Jetson Orin）

```
PyTorch → ONNX → TensorRT INT8 引擎 → ONNX Runtime（TensorRT 后端）
```

| 模型 | 精度 | Jetson Orin 延迟 |
|---|---|---|
| DINOv2-small + VQ-GAN | INT8 | ~10ms/帧 |
| WavTokenizer | FP16 | ~5ms/块 |
| VQ-VAE-IMU | INT8 | ~1ms/窗口 |
| VQ-VAE-TAC | INT8 | ~2ms/窗口 |
| 反射引擎 | — | <1ms |
| Qwen2.5-7B（INT4，llama.cpp） | INT4 | ~300ms/次 |

第一层 + 第二层合计 <20ms 完成；第三层异步运行，300ms 为一个推理周期。

---

## 实施路线图

### 已完成（代码框架）

| 模块 | 文件 | 状态 |
|---|---|---|
| VQ 量化器 | `sensorium/core/quantizer.py` | ✅ |
| 训练损失函数 | `sensorium/core/losses.py` | ✅ |
| 视觉编码器 | `sensorium/encoders/visual.py` | ✅ |
| 听觉编码器 | `sensorium/encoders/audio.py` | ✅ |
| IMU 编码器 | `sensorium/encoders/imu.py` | ✅ |
| 触觉编码器 | `sensorium/encoders/tactile.py` | ✅ |
| 摄像头驱动 | `sensorium/drivers/camera.py` | ✅ |
| 麦克风驱动 | `sensorium/drivers/microphone.py` | ✅ |
| IMU 驱动 | `sensorium/drivers/imu.py` | ✅ |
| FSR 触觉驱动 | `sensorium/drivers/tactile.py` | ✅ |
| 反射引擎 | `sensorium/reflex/engine.py` | ✅ |
| 主运行时管道 | `sensorium/pipeline/runtime.py` | ✅ |
| LLM 词汇表扩展 | `sensorium/llm/vocab.py` | ✅ |
| LLM 上下文窗口 | `sensorium/llm/context.py` | ✅ |
| LLM 推理进程 | `sensorium/llm/process.py` | ✅ |
| 视觉训练脚本 | `train/train_visual.py` | ✅ |
| IMU 训练脚本 | `train/train_imu.py` | ✅ |
| 触觉训练脚本 | `train/train_tactile.py` | ✅ |
| 跨模态对齐训练 | `train/train_alignment.py` | ✅ |
| LLM 微调脚本 | `train/train_llm.py` | ✅ |
| 同步录制工具 | `data/collector/record.py` | ✅ |
| 训练语料生成 | `data/preprocessor/build_dataset.py` | ✅ |
| Arduino FSR 固件 | `hardware/tactile_array/tactile_array.ino` | ✅ |
| 码本检查工具 | `tools/inspect_codebook.py` | ✅ |
| TensorRT 导出 | `deploy/jetson/export.py` | ✅ |

### 待办（按优先级）

#### 阶段一：硬件就位（当前阻塞点）

- [ ] 采购 FSR 传感器（约 80 个）、Arduino Mega、分压电阻
- [ ] 焊接 / 布线 FSR 阵列，按 `hardware/tactile_array/` 接线
- [ ] 刷入 Arduino 固件，验证串口帧格式
- [ ] 测试每路传感器驱动（摄像头 / 麦克风 / IMU / 触觉逐一验通）

#### 阶段二：数据采集 + Stage 1 训练

- [ ] 运行 `data/collector/record.py` 累计录制数据（目标：每模态 50 小时以上）
- [ ] 运行 `build_dataset.py --mode stage1` 生成训练数据
- [ ] 训练 IMU 编码器，验证码本利用率 > 80%
- [ ] 训练触觉编码器，验证码本利用率 > 80%
- [ ] 训练视觉编码器
- [ ] 运行 `inspect_codebook.py`，填入反射引擎的 `atom_id`
- [ ] **里程碑**：反射层跑通，被摸头 → 触发动作，听到声音 → 转头

#### 阶段三：对齐 + LLM + 部署

- [ ] 运行 `build_dataset.py --mode stage2` 生成对齐数据
- [ ] 训练跨模态对齐（Stage 2）
- [ ] 用 Whisper ASR 生成 LLM 训练语料
- [ ] LLM 微调 Step A（Embedding 对齐）
- [ ] LLM 微调 Step B（LoRA 全量微调）
- [ ] 在 Jetson Orin 上运行 `deploy/jetson/export.py` 导出 TensorRT INT8
- [ ] 端到端联调：`run.py` + `run_llm.py` 同时启动
- [ ] **里程碑**：系统理解"用户在被摸头时说了什么"并作出上下文相关的回应

---

## 核心开源依赖

| 组件 | 仓库 | 说明 |
|---|---|---|
| 视觉骨干 | [DINOv2](https://github.com/facebookresearch/dinov2) | 冻结使用 |
| 视觉量化 | [VQGAN-pytorch](https://github.com/dome272/VQGAN-pytorch) | 只训练头部 |
| 听觉 Token 化 | [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) | 直接使用，ICLR 2025 |
| 跨模态锚点 | [ImageBind](https://github.com/facebookresearch/ImageBind) | Stage 2 对齐信号 |
| 触觉架构参考 | [Sparsh](https://github.com/facebookresearch/sparsh) | Meta，ICLR 2025 |
| IMU 架构参考 | [IMU-Video-MAE](https://github.com/mf-zhang/IMU-Video-MAE) | ECCV 2024 |
| LLM 微调 | [PEFT](https://github.com/huggingface/peft) | LoRA 训练 |
| LLM 推理 | [llama.cpp](https://github.com/ggerganov/llama.cpp) | INT4 边缘推理 |
| ASR | [Whisper](https://github.com/openai/whisper) | 语料生成用 |
| 进程通信 | [ZeroMQ](https://zeromq.org/) | 替代 ROS2，轻量级跨进程 |

---

## 仓库结构

```
sensorium/
├── hardware/
│   └── tactile_array/
│       └── tactile_array.ino     # Arduino FSR 阵列固件
│
├── data/
│   ├── collector/
│   │   └── record.py             # 多模态同步录制工具
│   └── preprocessor/
│       └── build_dataset.py      # Stage 1/2/3 训练数据生成
│
├── sensorium/                    # 核心 Python 包
│   ├── core/
│   │   ├── quantizer.py          # VQ 量化器（EMA + 死码重置）
│   │   └── losses.py             # 重建 / 时序预测 / 跨模态对齐损失
│   ├── encoders/
│   │   ├── visual.py             # DINOv2 + VQ-GAN
│   │   ├── audio.py              # WavTokenizer 封装
│   │   ├── imu.py                # 六轴 IMU VQ-VAE
│   │   └── tactile.py            # 时空压力图 VQ-VAE
│   ├── drivers/
│   │   ├── camera.py             # OpenCV 摄像头（独立线程）
│   │   ├── microphone.py         # sounddevice 麦克风
│   │   ├── imu.py                # I2C IMU（smbus2）
│   │   └── tactile.py            # Arduino 串口 FSR 阵列
│   ├── reflex/
│   │   └── engine.py             # 反射规则引擎（<50ms）
│   ├── pipeline/
│   │   └── runtime.py            # asyncio 主运行时 + ZMQ
│   └── llm/
│       ├── vocab.py              # LLM 感官词汇表扩展
│       ├── context.py            # 滚动上下文窗口
│       └── process.py            # LLM 独立推理进程
│
├── train/
│   ├── train_visual.py           # Stage 1：视觉编码器
│   ├── train_imu.py              # Stage 1：IMU 编码器
│   ├── train_tactile.py          # Stage 1：触觉编码器
│   ├── train_alignment.py        # Stage 2：跨模态对齐
│   └── train_llm.py              # Stage 3：LLM 微调（Step A + B）
│
├── configs/
│   ├── visual.yaml
│   ├── imu.yaml
│   ├── tactile.yaml
│   ├── alignment.yaml
│   └── llm.yaml
│
├── tools/
│   └── inspect_codebook.py       # 码本检查工具（填写反射规则用）
│
├── deploy/
│   └── jetson/
│       └── export.py             # ONNX → TensorRT INT8 导出
│
├── run.py                        # 主感知进程入口
└── run_llm.py                    # LLM 推理进程入口
```

---

## 关于命名

**Sensorium**（拉丁语）——神经科学术语，指生物体感知世界的全部感官系统的总和：所有感官输入、处理它们的大脑结构，以及它们共同构建出的统一感知世界。

这正是这个项目想做的事。

---

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"

# 录制数据
python data/collector/record.py --duration 3600

# Stage 1 训练（三路独立，可并行）
python train/train_imu.py      data_dir=data/processed/imu_recordings/
python train/train_tactile.py  data_dir=data/processed/tactile_recordings/
python train/train_visual.py   data_dir=data/processed/video_recordings/

# 检查码本，填写反射规则
python tools/inspect_codebook.py --encoder imu     --checkpoint checkpoints/imu/best.ckpt     --mode interactive
python tools/inspect_codebook.py --encoder tactile --checkpoint checkpoints/tactile/best.ckpt --mode interactive

# Stage 2 + Stage 3
python train/train_alignment.py
python train/train_llm.py step=AB model_path=Qwen/Qwen2.5-7B-Instruct

# 运行系统（两个终端）
python run.py     --imu-checkpoint checkpoints/imu/best.ckpt --tactile-checkpoint checkpoints/tactile/best.ckpt
python run_llm.py --model models/qwen2.5-7b-q4_k_m.gguf --backend llama_cpp
```

## 当前状态

代码框架完整，等待硬件就位后进入数据采集阶段。  
当前阻塞点：FSR 触觉阵列的硬件采购与组装。
