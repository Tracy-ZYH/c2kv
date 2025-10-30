import json
from dataclasses import dataclass, field, asdict
from transformers.training_args import TrainingArguments
from typing import Optional, List, Tuple, Union, Dict


@dataclass
class ModelArgs:
    model_cache_dir: str = field(
        default=None,
        metadata={'help': 'Default path to save language models.'}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    padding_side: str = field(
        default='right',
        metadata={'help': 'Padding side of the tokenizer.'}
    )
    attn_impl: Optional[str] = field(
        default="sdpa",
        metadata={'help': 'Attention implementation.'}
    )
    max_length: int = field(
        default=4096,
        metadata={'help': 'Maximum length of the input.'}
    )
    chat_template: str = field(
        default='hf',
        metadata={'help': 'Instruction template name in fastchat.'}
    )
    max_position_embeddings: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum position.'}
    )
    rope_theta: Optional[float] = field(
        default=None,
        metadata={'help': 'RoPE base (theta).'}
    )
    rope_method: Optional[str] = field(
        default=None,
        metadata={'help': 'How to scale RoPE? {linear, dynamic, yarn}'},
    )
    rope_factor: float = field(
        default=1.,
        metadata={'help': 'RoPE scaling factor.'},
    )
    dtype: str = field(
        default="bf16",
        metadata={'help': 'Data type for embeddings.'}
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={'help': 'Device map for loading the model. Set to auto to load across devices.'}
    )
    batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'},
    )
    enable_gist: bool = field(
        default=False,
        metadata={'help': 'Use Gist?'}
    )
    gist_type: str = field(
        default="interleave-4",
        metadata={'help': 'Gist type.'}
    )
    gist_mode: str = field(
        default="1024-16",
        metadata={'help': 'Gist Chunk Size and Max Chunk Number.'}
    )
    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={'help': 'How many tokens at maximum to return?'},
    )
    do_sample: Optional[bool] = field(
        default=None,
        metadata={'help': 'Do sampling when decoding?'},
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={'help': 'Sampling temperature.'},
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={'help': "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."}
    )

    enable_tp: bool = field(
        default=False,
        metadata={'help': 'Use tensor parallel to wrap the model?'}
    )

    lora: Optional[str] = field(
        default=None,
        metadata={'help': 'LoRA ID.'},
    )
    lora_unload: bool = field(
        default=True,
        metadata={'help': 'Merge and unload LoRA?'},
    )
    
    def get_generation_config(self):
        generation_config = {}
        if self.max_new_tokens is not None:
            generation_config["max_new_tokens"] = self.max_new_tokens
        if self.do_sample is not None:
            generation_config["do_sample"] = self.do_sample
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.top_p is not None:
            generation_config["top_p"] = self.top_p
        return generation_config

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)


@dataclass
class TrainingArgs(TrainingArguments):
    pretrain_min_length: int = field(
        default=1024,
        metadata={'help': 'Minimum length of the input for training'}
    )
    pretrain_max_length: int = field(
        default=4096,
        metadata={'help': 'Maximum length of the input for training'}
    )
    only_train_gist: bool = field(
        default=True,
        metadata={'help': 'Only train gist?'}
    )
    train_data: str = field(
        default=None,
        metadata={'help': 'Path to training data'}
    )
