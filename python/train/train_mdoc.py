import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from transformers import HfArgumentParser, DataCollatorWithPadding

from .train_data import get_dataset
from .trainer import GistMultiDocTrainer
from models import *
from gist_args import ModelArgs, TrainingArgs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class MDocDataArgs:
    device_type: str = field(
        default="auto",
        metadata={"help": "Training device type: auto, cuda, npu, or cpu."},
    )
    npu_attn_impl: str = field(
        default="npu_fusion_attention",
        metadata={"help": "Attention backend to use when device_type resolves to npu."},
    )
    train_data_cleaned: str | None = field(
        default=None,
        metadata={"help": "Optional root containing cleaned mdoc datasets."},
    )
    train_sources: str = field(
        default="hotpotqa,wikimqa,longmagpie",
        metadata={"help": "Comma-separated mdoc training sources."},
    )
    eval_sources: str = field(
        default="hotpotqa,wikimqa",
        metadata={"help": "Comma-separated mdoc eval sources."},
    )
    train_source_sizes: str | None = field(
        default=None,
        metadata={"help": "Optional comma-separated per-source sample caps."},
    )
    eval_num_samples: int = field(default=512)
    max_doc_length: int = field(default=1024)
    max_doc_num: int = field(default=10)
    max_length: int = field(default=1024)
    max_system_length: int = field(default=256)
    dynamic_context_cap: int = field(default=4096)
    dataset_num_proc: int = field(default=8)
    dataset_load_from_cache_file: bool = field(default=True)
    preprocess_cache_version: str = field(default="mdoc_answer_preserve_v2")


def _resolve_device_type(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch_npu  # noqa: F401
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _setup_device(model_args, data_args: MDocDataArgs) -> str:
    device_type = _resolve_device_type(data_args.device_type)
    resolved_device = device_type
    if device_type == "npu":
        import torch
        import torch_npu  # noqa: F401

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible_count = None
        if hasattr(torch, "npu"):
            try:
                visible_count = torch.npu.device_count()
            except Exception:
                visible_count = None
        if visible_count is not None and local_rank >= visible_count:
            visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("ASCEND_VISIBLE_DEVICES")
            raise ValueError(
                f"LOCAL_RANK={local_rank} but only {visible_count} NPU device(s) are visible "
                f"({visible}). Set NPROC_PER_NODE={visible_count}, or expose more NPUs."
            )
        if hasattr(torch, "npu"):
            torch.npu.set_device(local_rank)
            resolved_device = f"npu:{local_rank}"
            logger.info(
                "Using NPU local_rank=%s current_device=%s visible_count=%s visible=%s",
                local_rank,
                torch.npu.current_device(),
                visible_count,
                os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("ASCEND_VISIBLE_DEVICES"),
            )
        if model_args.attn_impl in (None, "flex_attention", "flash_attention_2"):
            logger.info(
                "Overriding attn_impl=%s to npu_attn_impl=%s for NPU training",
                model_args.attn_impl,
                data_args.npu_attn_impl,
            )
            model_args.attn_impl = data_args.npu_attn_impl
        if model_args.device_map == "auto":
            logger.info("Disabling device_map=auto for NPU distributed/deepspeed training")
            model_args.device_map = None
    return resolved_device


def _split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _parse_source_sizes(value: str | None, num_sources: int) -> list[int | None]:
    if value is None or not value.strip():
        return [None] * num_sources
    parsed: list[int | None] = []
    for item in _split_csv(value):
        parsed.append(None if item.lower() in {"none", "all", "-1"} else int(item))
    if len(parsed) != num_sources:
        raise ValueError(f"train_source_sizes has {len(parsed)} values but train_sources has {num_sources}")
    return parsed


def _candidate_roots(train_data: str, train_data_cleaned: str | None) -> tuple[Path, Path]:
    train_root = Path(train_data)
    if train_data_cleaned is not None:
        cleaned_root = Path(train_data_cleaned)
    elif (train_root / "hotpotqa_train_cleaned").exists() or (train_root / "wikimqa_train_cleaned").exists():
        cleaned_root = train_root
    else:
        cleaned_root = Path(str(train_root) + "_cleaned")
    return train_root, cleaned_root


def _source_path(source: str, train_root: Path, cleaned_root: Path) -> str:
    source = source.lower()
    candidates: dict[str, list[Path | str]] = {
        "hotpotqa": [cleaned_root / "hotpotqa_train_cleaned", train_root / "hotpotqa_train_cleaned"],
        "wikimqa": [cleaned_root / "wikimqa_train_cleaned", train_root / "wikimqa_train_cleaned"],
        "longmagpie": [cleaned_root / "longmagpie_cleaned", train_root / "longmagpie_cleaned", train_root / "longmagpie_1024"],
        "tulu3": ["allenai/tulu-3-sft-mixture"],
        "nextcoder": [train_root / "microsoft--NextCoderDataset"],
    }
    if source not in candidates:
        raise ValueError(f"Unsupported train source: {source}")
    for candidate in candidates[source]:
        if isinstance(candidate, str) or candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find source {source!r}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates[source])
    )


def _load_mdoc_mix(tokenizer, training_args, data_args: MDocDataArgs):
    train_root, cleaned_root = _candidate_roots(training_args.train_data, data_args.train_data_cleaned)
    dataset_args = {
        "tokenizer": tokenizer,
        "shuffle_seed": training_args.dataset_shuffle_seed,
        "max_doc_length": data_args.max_doc_length,
        "max_doc_num": data_args.max_doc_num,
        "max_length": data_args.max_length,
        "max_system_length": data_args.max_system_length,
        "dynamic_context_cap": data_args.dynamic_context_cap,
        "dataset_num_proc": data_args.dataset_num_proc,
        "dataset_load_from_cache_file": data_args.dataset_load_from_cache_file,
        "preprocess_cache_version": data_args.preprocess_cache_version,
    }

    train_sources = _split_csv(data_args.train_sources)
    train_sizes = _parse_source_sizes(data_args.train_source_sizes, len(train_sources))
    if not train_sources:
        raise ValueError("train_sources must not be empty")

    train_datasets = []
    for source, num_samples in zip(train_sources, train_sizes):
        path = _source_path(source, train_root, cleaned_root)
        logger.info("Loading train source %s from %s with num_samples=%s", source, path, num_samples)
        train_datasets.append(get_dataset("mdoc", path, num_samples=num_samples, **dataset_args))
    train_dataset = train_datasets[0]
    if len(train_datasets) > 1:
        train_dataset.merge(train_datasets[1:])

    eval_sources = _split_csv(data_args.eval_sources)
    eval_datasets = []
    for source in eval_sources:
        path = _source_path(source, train_root, cleaned_root)
        logger.info("Loading eval source %s from %s with num_samples=%s", source, path, data_args.eval_num_samples)
        eval_datasets.append(get_dataset("mdoc_eval", path, num_samples=data_args.eval_num_samples, **dataset_args))
    eval_dataset = eval_datasets[0] if eval_datasets else None
    if eval_dataset is not None and len(eval_datasets) > 1:
        eval_dataset.merge(eval_datasets[1:], method="concat")
    return train_dataset, eval_dataset


def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs, MDocDataArgs])
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    device_type = _setup_device(model_args, data_args)

    if model_args.gist_gradient_checkpointing:
        import models.gist_utils as _gist_utils
        _gist_utils.GIST_GRADIENT_CHECKPOINTING = True

    model, tokenizer = get_model_and_tokenizer(
        model_args,
        device=device_type,
        evaluation_mode=not training_args.do_train,
    )

    if model_args.enable_gist and training_args.only_train_gist:
        for name, param in model.named_parameters():
            param.requires_grad_('gist' in name)
    
    logger.info(f"Total Model params: {format_numel_str(sum(p.numel() for p in model.parameters()))}")
    logger.info(f"Trainable Model params: {format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}")

    with training_args.main_process_first(desc="Get dataset"):
        train_dataset, eval_dataset = _load_mdoc_mix(tokenizer, training_args, data_args)
        logger.info("Prepared train samples: %d", len(train_dataset))
        if eval_dataset is not None:
            logger.info("Prepared eval samples: %d", len(eval_dataset))

    trainer = GistMultiDocTrainer(
        model=model,
        args=training_args,
        max_doc_length=train_dataset.max_doc_length,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer, 
            padding=True,
            return_tensors='pt',
        ),
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        eval_result = trainer.evaluate()
        with training_args.main_process_first(desc="Evaluate model"):
            logger.info(f"Evaluation result: {eval_result}")
    
if __name__ == "__main__":
    main()
