import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import DataCollatorWithPadding, HfArgumentParser

from .train_data_multiturn import get_compress_history_dataset
from .trainer import GistMultiDocTrainer
from gist_args import ModelArgs, TrainingArgs
from models import *


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class CompressHistoryDataArgs:
    source_type: str = "jsonl"
    eval_data: Optional[str] = None
    max_doc_length: int = 1024
    min_doc_num: int = 2
    max_doc_num: int = 10
    max_length: int = 1024
    max_system_length: int = 4096
    history_selection: str = "tail"
    num_samples: Optional[int] = None
    eval_num_samples: Optional[int] = 512
    resolved_only: bool = True
    languages: Optional[str] = None
    max_total_chars: Optional[int] = None
    max_answer_chars: Optional[int] = None
    recent_message_num: int = 1
    max_samples_per_trace: Optional[int] = None
    num_proc: int = 8
    split_manifest_file: Optional[str] = None
    split_manifest_name: str = "subset_disjoint"
    eval_ratio: float = 0.1
    split_seed: int = 42
    max_samples_per_session: Optional[int] = 4
    require_tool_call: bool = False
    include_tools: bool = False
    max_input_chars: Optional[int] = None
    prefix_history_doc_num: Optional[int] = None
    prefix_history_exact: bool = False
    full_history_doc_num: int = 0
    split_oversized_history_docs: bool = True
    device_type: str = "auto"
    npu_attn_impl: str = "npu_fusion_attention"


def _resolve_device_type(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _visible_npu_count() -> Optional[int]:
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("ASCEND_VISIBLE_DEVICES")
    if visible_devices:
        return len([item for item in visible_devices.split(",") if item.strip()])
    try:
        import torch_npu  # noqa: F401

        return int(torch.npu.device_count())
    except Exception:
        return None


def _setup_device(model_args: ModelArgs, data_args: CompressHistoryDataArgs) -> str:
    device_type = _resolve_device_type(data_args.device_type)
    if device_type == "npu":
        import torch_npu  # noqa: F401

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible_count = _visible_npu_count()
        if visible_count is not None and local_rank >= visible_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {visible_count} NPU device(s) are visible. "
                "Check ASCEND_RT_VISIBLE_DEVICES and --nproc_per_node."
            )
        torch.npu.set_device(local_rank)
        if model_args.attn_impl in (None, "flex_attention", "flash_attention_2"):
            logger.info(
                "Overriding attn_impl=%s to npu_attn_impl=%s for NPU training",
                model_args.attn_impl,
                data_args.npu_attn_impl,
            )
            model_args.attn_impl = data_args.npu_attn_impl
    return device_type


def _source_kwargs(data_args: CompressHistoryDataArgs, split: str) -> Optional[dict]:
    if data_args.source_type == "open_swe":
        return {
            "resolved_only": data_args.resolved_only,
            "languages": data_args.languages,
            "max_total_chars": data_args.max_total_chars,
            "max_answer_chars": data_args.max_answer_chars,
            "recent_message_num": data_args.recent_message_num,
            "max_samples_per_trace": data_args.max_samples_per_trace,
            "num_proc": data_args.num_proc,
        }
    if data_args.source_type == "agent_llm_traces":
        sample_limit = data_args.num_samples if split == "train" else data_args.eval_num_samples
        return {
            "split": split,
            "eval_ratio": data_args.eval_ratio,
            "split_seed": data_args.split_seed,
            "split_manifest_file": data_args.split_manifest_file,
            "split_manifest_name": data_args.split_manifest_name,
            "max_samples_per_session": data_args.max_samples_per_session,
            "max_records": sample_limit,
            "require_tool_call": data_args.require_tool_call,
            "include_tools": data_args.include_tools,
            "max_input_chars": data_args.max_input_chars,
            "max_answer_chars": data_args.max_answer_chars,
            "prefix_history_doc_num": data_args.prefix_history_doc_num,
            "prefix_history_exact": data_args.prefix_history_exact,
        }
    return None


def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs, CompressHistoryDataArgs])
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
            param.requires_grad_("gist" in name)

    logger.info(
        f"Total Model params: {format_numel_str(sum(p.numel() for p in model.parameters()))}"
    )
    logger.info(
        "Trainable Model params: "
        f"{format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}"
    )

    with training_args.main_process_first(desc="Get compress-history dataset"):
        train_source_kwargs = _source_kwargs(data_args, "train")
        dataset_kwargs = {
            "source_type": data_args.source_type,
            "source_kwargs": train_source_kwargs,
            "max_doc_length": data_args.max_doc_length,
            "min_doc_num": data_args.min_doc_num,
            "max_doc_num": data_args.max_doc_num,
            "max_length": data_args.max_length,
            "max_system_length": data_args.max_system_length,
            "history_selection": data_args.history_selection,
            "full_history_doc_num": data_args.full_history_doc_num,
            "split_oversized_history_docs": data_args.split_oversized_history_docs,
            "num_samples": data_args.num_samples,
            "shuffle_seed": training_args.dataset_shuffle_seed,
            "num_proc": data_args.num_proc,
        }
        train_dataset = get_compress_history_dataset(
            training_args.train_data,
            tokenizer=tokenizer,
            **dataset_kwargs,
        )
        eval_dataset = None
        eval_strategy = getattr(training_args, "eval_strategy", None)
        if eval_strategy is None:
            eval_strategy = getattr(training_args, "evaluation_strategy", "no")
        should_build_eval = (
            not training_args.do_train
            or bool(getattr(training_args, "do_eval", False))
            or str(eval_strategy).lower() != "no"
        )
        eval_data = data_args.eval_data
        if should_build_eval and data_args.source_type == "agent_llm_traces" and eval_data is None:
            eval_data = training_args.train_data
        if should_build_eval and eval_data:
            eval_kwargs = dict(dataset_kwargs)
            eval_kwargs["num_samples"] = data_args.eval_num_samples
            eval_kwargs["shuffle_seed"] = 42
            eval_kwargs["source_kwargs"] = _source_kwargs(data_args, "eval")
            eval_dataset = get_compress_history_dataset(
                eval_data,
                tokenizer=tokenizer,
                **eval_kwargs,
            )

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
            return_tensors="pt",
        ),
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        if eval_dataset is None:
            raise ValueError("--eval_data is required when running evaluation only")
        eval_result = trainer.evaluate()
        with training_args.main_process_first(desc="Evaluate model"):
            logger.info(f"Evaluation result: {eval_result}")


if __name__ == "__main__":
    main()
