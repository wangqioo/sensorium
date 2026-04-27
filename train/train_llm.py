"""
LLM 多模态微调脚本 — Stage 3

两步训练：
  Step A：只训练新增感官 token 的 embedding（LLM 主体冻结）
           目标：让感官 token 的嵌入向量融入原始语言空间
           时长：约 1 天（A100）

  Step B：LoRA 全量微调（解冻，rank=64）
           目标：让 LLM 学会从感官 token 序列理解场景
           时长：约 1 周（A100）

训练数据格式（多模态交织序列的 JSONL 文件）：
  每行一个 JSON 对象：
  {
    "vis_tokens":  [437, 201, 333],    ← VIS atom IDs
    "aud_tokens":  [89, 156],          ← AUD atom IDs
    "imu_tokens":  [12],               ← IMU atom IDs
    "tac_tokens":  [87, 91],           ← TAC atom IDs
    "text":        "你好，今天好累啊"   ← 对应的文字（ASR 转写或手动标注）
  }

运行：
  # Step A
  python train/train_llm.py \
    model_path=Qwen/Qwen2.5-7B-Instruct \
    data_path=data/multimodal_sequences.jsonl \
    step=A

  # Step B（Step A 完成后）
  python train/train_llm.py \
    model_path=checkpoints/llm/step_a/final \
    data_path=data/multimodal_sequences.jsonl \
    step=B
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

import hydra
from omegaconf import DictConfig, OmegaConf

from sensorium.llm.vocab import SensoriumTokenizer


# ——— 数据集 ———

class MultiModalSequenceDataset(Dataset):
    """
    读取 JSONL 格式的多模态训练序列，转为 LLM 训练用的 input_ids。

    每行格式：
      {"vis_tokens": [...], "aud_tokens": [...], "imu_tokens": [...],
       "tac_tokens": [...], "text": "..."}

    输出：
      {"input_ids": Tensor, "labels": Tensor, "attention_mask": Tensor}

    Labels 设计：
      感官 token 位置：labels = -100（不参与 loss，只让 LLM 学预测文字）
      文字 token 位置：labels = input_ids（正常 next-token prediction）
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: SensoriumTokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"[MultiModalDataset] {len(self.samples)} 条训练样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]

        # 感官部分 token ID（通过 SensoriumTokenizer 编码）
        sensor_ids = self.tokenizer.encode_multimodal(
            vis_tokens=s.get("vis_tokens"),
            aud_tokens=s.get("aud_tokens"),
            imu_tokens=s.get("imu_tokens"),
            tac_tokens=s.get("tac_tokens"),
        )

        # 文字部分 token ID
        text_ids = self.tokenizer.tokenizer.encode(
            s.get("text", ""),
            add_special_tokens=False,
        )
        # 加上 EOS，让模型知道序列结束
        eos = self.tokenizer.tokenizer.eos_token_id
        if eos:
            text_ids = text_ids + [eos]

        input_ids = sensor_ids + text_ids

        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]

        # Labels：感官 token 不参与 loss（设为 -100），文字 token 正常
        n_sensor = min(len(sensor_ids), self.max_length)
        n_text   = len(input_ids) - n_sensor
        labels   = [-100] * n_sensor + input_ids[n_sensor:]

        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        labels_t    = torch.tensor(labels,    dtype=torch.long)
        attn_mask   = torch.ones_like(input_ids_t)

        return {
            "input_ids":      input_ids_t,
            "labels":         labels_t,
            "attention_mask": attn_mask,
        }


def collate_fn(batch: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """动态 padding 到 batch 内最长序列。"""
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids  = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels     = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn_mask  = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, x in enumerate(batch):
        L = x["input_ids"].size(0)
        input_ids[i, :L]  = x["input_ids"]
        labels[i, :L]     = x["labels"]
        attn_mask[i, :L]  = x["attention_mask"]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}


# ——— Step A：只训练 embedding ———

def run_step_a(cfg: DictConfig, tokenizer: SensoriumTokenizer) -> str:
    """
    冻结 LLM 全部参数，只让新增感官 token 的 embedding 参与训练。
    """
    print("\n=== Step A：感官 Token Embedding 对齐 ===")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 扩展 embedding 层以容纳新 token
    model.resize_token_embeddings(len(tokenizer))

    # 冻结除 embedding 层以外的所有参数
    for name, param in model.named_parameters():
        param.requires_grad = "embed" in name.lower()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数：{n_trainable:,}（embedding 层）")

    dataset = MultiModalSequenceDataset(cfg.data_path, tokenizer, cfg.max_length)

    args = TrainingArguments(
        output_dir="checkpoints/llm/step_a",
        num_train_epochs=cfg.step_a.epochs,
        per_device_train_batch_size=cfg.step_a.batch_size,
        gradient_accumulation_steps=cfg.step_a.grad_accum,
        learning_rate=cfg.step_a.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda b: collate_fn(b, tokenizer.tokenizer.pad_token_id or 0),
    )
    trainer.train()

    output_path = "checkpoints/llm/step_a/final"
    model.save_pretrained(output_path)
    tokenizer.tokenizer.save_pretrained(output_path)
    print(f"Step A 完成，保存至：{output_path}")
    return output_path


# ——— Step B：LoRA 全量微调 ———

def run_step_b(cfg: DictConfig, tokenizer: SensoriumTokenizer) -> str:
    """
    在 Step A 的基础上，用 LoRA 对整个模型做全量微调。
    """
    print("\n=== Step B：LoRA 全量微调 ===")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    # LoRA 配置
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.step_b.lora_rank,
        lora_alpha=cfg.step_b.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = MultiModalSequenceDataset(cfg.data_path, tokenizer, cfg.max_length)

    args = TrainingArguments(
        output_dir="checkpoints/llm/step_b",
        num_train_epochs=cfg.step_b.epochs,
        per_device_train_batch_size=cfg.step_b.batch_size,
        gradient_accumulation_steps=cfg.step_b.grad_accum,
        learning_rate=cfg.step_b.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,  # 节省显存
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda b: collate_fn(b, tokenizer.tokenizer.pad_token_id or 0),
    )
    trainer.train()

    output_path = "checkpoints/llm/step_b/final"
    model.save_pretrained(output_path)
    tokenizer.tokenizer.save_pretrained(output_path)
    print(f"Step B 完成，保存至：{output_path}")
    return output_path


# ——— Hydra 入口 ———

@hydra.main(config_path="../configs", config_name="llm", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    tokenizer = SensoriumTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    print(f"词汇表大小：{len(tokenizer):,}（含 {len(tokenizer) - 151936} 个感官 token）")

    if cfg.step == "A":
        run_step_a(cfg, tokenizer)
    elif cfg.step == "B":
        run_step_b(cfg, tokenizer)
    elif cfg.step == "AB":
        path_a = run_step_a(cfg, tokenizer)
        cfg.model_path = path_a       # Step B 从 Step A 的输出继续
        run_step_b(cfg, tokenizer)
    else:
        raise ValueError(f"step 必须是 A / B / AB，实际是：{cfg.step}")


if __name__ == "__main__":
    main()
