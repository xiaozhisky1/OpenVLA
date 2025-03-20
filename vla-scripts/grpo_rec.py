# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from grpo_trainer import OpenVLAGRPOTrainer
from grpo_config import GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments, AutoProcessor
import yaml
import json
import random
import math


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

'''
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.5, the reward is 1.0, otherwise 0.0 .
    This is a hard reward, maybe the soft reward is better and could be used in the future .
'''
def iou_reward(completions, action, **kwargs):
    def dis(action1, action2):
        action1 = np.array(action1)
        action2 = np.array(action2)
        euclidean_distance = np.linalg.norm(action1 - action2)
        return float(euclidean_distance)
    contents = [completion[0]["value"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    action_pattern =r'\[(?:[01]?\d{1,2}|2[0-4][0-9]|255)(?:,(?:[01]?\d{1,2}|2[0-4][0-9]|255)){6}\]'
    for content, act in zip(contents, action):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                action_match = re.search(action_pattern, content_answer)
                if action_match:
                    action_pred = [int(action_match.group(1)), int(action_match.group(2)), int(action_match.group(3)), int(action_match.group(4)), int(action_match.group(5)),  int(action_match.group(6)),  int(action_match.group(7))]
                    # 阈值有待商榷
                    if dis(action_pred, action) > 5:
                        reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {action_pred}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
    completion_contents = [completion[0]["value"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": iou_reward,
    "format": format_reward,
}

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


def main(script_args, training_args, model_args, cfg: FinetuneConfig):

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState() # 初始化 分布式训练的状态 PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory 生成实验 ID 及日志目录
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning 配置量化（暂不使用）
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # 将 OpenVLA 注册为 Hugging Face 的 AutoModel，方便后续调用 from_pretrained() 直接加载。
    # AutoConfig.register("openvla", OpenVLAConfig)
    # AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    # AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    # AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # 加载 OpenVLA 处理器 & 模型
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True) # processor：用于处理 文本 & 图像输入
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ) # vla：加载 OpenVLA 模型，设置 torch_dtype=torch.bfloat16 以减少显存占用

    # 设备分配 & LoRA 微调 Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    lora_config = None
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        # vla = get_peft_model(vla, lora_config)
        # vla.print_trainable_parameters()

    # 使用 torch.nn.parallel.DistributedDataParallel (DDP) 进行 多 GPU 训练  Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    # vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    # trainable_params = [param for param in vla.parameters() if param.requires_grad] # 仅优化 需要更新的参数（冻结其他层
    # optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # ---------Create Action Tokenizer---------
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    # 加载数据集 使用 RLDS 格式（强化学习数据）
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug, # 使用数据增强
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    # 用于保存数据集的统计信息，这些信息在推理（inference）时用于反归一化动作（de-normalize actions）
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader 创建数据整理器, 确保训练过程中数据格式的一致性
    # 由于动作序列的长度不同，我们需要填充（Padding）或截断，以保证所有输入的 input_ids 具有相同长度，方便 批量处理（batch processing）
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Load the dataset
    # dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    trainer_cls = OpenVLAGRPOTrainer
    # Initialize the GRPO trainer

    trainer = trainer_cls(
        model=cfg.vla_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=vla_dataset,
        eval_dataset=None,
        peft_config=lora_config,
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
        data_collator=collator,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
