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

    if model_args.gist_gradient_checkpointing:
        import models.gist_utils as _gist_utils
        _gist_utils.GIST_GRADIENT_CHECKPOINTING = True

    model, tokenizer = get_model_and_tokenizer(model_args, evaluation_mode=not training_args.do_train)

    if model_args.enable_gist and training_args.only_train_gist:
        for name, param in model.named_parameters():
            param.requires_grad_('gist' in name)
    
    logger.info(f"Total Model params: {format_numel_str(sum(p.numel() for p in model.parameters()))}")
    logger.info(f"Trainable Model params: {format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}")

    with training_args.main_process_first(desc="Get dataset"):
        dataset_args = {
            'tokenizer': tokenizer, 'shuffle_seed': training_args.dataset_shuffle_seed,
            'max_doc_length': 1024, 'max_doc_num': 10, 'max_length': 1024,
        }
        hotpotqa_path = os.path.join(training_args.train_data + "_cleaned", "hotpotqa_train_cleaned")
        wikimqa_path = os.path.join(training_args.train_data + "_cleaned", "wikimqa_train_cleaned")
        longmagpie_path = os.path.join(training_args.train_data, "longmagpie_1024")
        nextcoder_path = os.path.join(training_args.train_data, "microsoft--NextCoderDataset")

        train_dataset = get_dataset('mdoc', hotpotqa_path, **dataset_args)
        wikimqa_train = get_dataset('mdoc', wikimqa_path, **dataset_args)
        tulu3_train = get_dataset('mdoc', "allenai/tulu-3-sft-mixture", **dataset_args)
        nextcoder_train = get_dataset('mdoc', nextcoder_path, **dataset_args)
        longmagpie_train = get_dataset('mdoc', longmagpie_path, **dataset_args)

        eval_dataset = get_dataset('mdoc_eval', hotpotqa_path, **dataset_args)
        wikimqa_eval = get_dataset('mdoc_eval', wikimqa_path, **dataset_args)

        train_dataset.data = train_dataset.data.select(range(40000))
        wikimqa_train.data = wikimqa_train.data.select(range(40000))
        tulu3_train.data = tulu3_train.data.select(range(80000))
        nextcoder_train.data = nextcoder_train.data.select(range(56000))
        longmagpie_train.data = longmagpie_train.data.select(range(40000))

        train_dataset.merge([wikimqa_train, tulu3_train, nextcoder_train, longmagpie_train])
        eval_dataset.merge([wikimqa_eval], method='concat')

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
