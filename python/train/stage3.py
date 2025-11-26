import os
import logging
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


def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = get_model_and_tokenizer(model_args, evaluation_mode=not training_args.do_train)

    if model_args.enable_gist and training_args.only_train_gist:
        for name, param in model.named_parameters():
            param.requires_grad_('gist' in name)
    
    logger.info(f"Total Model params: {format_numel_str(sum(p.numel() for p in model.parameters()))}")
    logger.info(f"Trainable Model params: {format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}")

    with training_args.main_process_first(desc="Get dataset"):
        dataset_args = {'tokenizer': tokenizer}
        train_dataset = get_dataset('mdoc', training_args.train_data, **dataset_args)
        eval_dataset = get_dataset('mdoc_eval', training_args.train_data, **dataset_args)

    system_ids = tokenizer(train_dataset.system_prompt_ids, return_tensors='pt')["input_ids"]

    trainer = GistMultiDocTrainer(
        model=model,
        args=training_args,
        system_ids=system_ids,
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
