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
        dataset_args = {
            'tokenizer': tokenizer, 'shuffle_seed': training_args.dataset_shuffle_seed,
            'max_doc_num': 10, 'max_doc_length': 1536,
        }
        hotpotqa_train = get_dataset('mdoc', "../datasets/hotpotqa_train.jsonl", **dataset_args)
        hotpotqa_eval = get_dataset('mdoc_eval', "../datasets/hotpotqa_train.jsonl", **dataset_args)
        longmagpie_path = os.path.join(training_args.train_data, "longmagpie_processed_long")
        longmagpie_train = get_dataset('mdoc', longmagpie_path, **dataset_args)
        longmagpie_eval = get_dataset('mdoc_eval', longmagpie_path, **dataset_args)
        longalpaca_path = os.path.join(training_args.train_data, "longalpaca_processed_long")
        longalpaca_train = get_dataset('mdoc', longalpaca_path, **dataset_args)
        longalpaca_eval = get_dataset('mdoc_eval', longalpaca_path, **dataset_args)
        # select 10000 samples each from longmagpie and hotpotqa
        hotpotqa_train.select(range(20000))
        longmagpie_train.select(range(20000))
        # print train and eval dataset lengths
        logger.info(f"Train dataset lengths: longmagpie: {len(longmagpie_train)}, longalpaca: {len(longalpaca_train)}, hotpotqa: {len(hotpotqa_train)}")
        logger.info(f"Eval dataset lengths: longmagpie: {len(longmagpie_eval)}, longalpaca: {len(longalpaca_eval)}, hotpotqa: {len(hotpotqa_eval)}")
        train_dataset = longmagpie_train.merge([hotpotqa_train, longalpaca_train, longalpaca_eval])
        eval_dataset = hotpotqa_eval

    trainer = GistMultiDocTrainer(
        model=model,
        args=training_args,
        system_ids=train_dataset.system_prompt_ids,
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
