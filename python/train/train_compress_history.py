import logging
from dataclasses import dataclass
from typing import Optional

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
    max_doc_num: int = 10
    max_length: int = 1024
    max_system_length: int = 2048
    history_selection: str = "tail"
    num_samples: Optional[int] = None
    eval_num_samples: Optional[int] = 512


def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs, CompressHistoryDataArgs])
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    if model_args.gist_gradient_checkpointing:
        import models.gist_utils as _gist_utils

        _gist_utils.GIST_GRADIENT_CHECKPOINTING = True

    model, tokenizer = get_model_and_tokenizer(
        model_args,
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
        dataset_kwargs = {
            "source_type": data_args.source_type,
            "max_doc_length": data_args.max_doc_length,
            "max_doc_num": data_args.max_doc_num,
            "max_length": data_args.max_length,
            "max_system_length": data_args.max_system_length,
            "history_selection": data_args.history_selection,
            "num_samples": data_args.num_samples,
            "shuffle_seed": training_args.dataset_shuffle_seed,
        }
        train_dataset = get_compress_history_dataset(
            training_args.train_data,
            tokenizer=tokenizer,
            **dataset_kwargs,
        )
        eval_dataset = None
        if data_args.eval_data:
            eval_kwargs = dict(dataset_kwargs)
            eval_kwargs["num_samples"] = data_args.eval_num_samples
            eval_kwargs["shuffle_seed"] = 42
            eval_dataset = get_compress_history_dataset(
                data_args.eval_data,
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
