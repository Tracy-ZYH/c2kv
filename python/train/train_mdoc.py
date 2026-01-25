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
            'max_doc_length': 1536, 'max_doc_num': 10,
        }
        # load mdoc QA datasets
        train_dataset = get_dataset('mdoc', "../datasets/hotpotqa_train.jsonl", **dataset_args)
        eval_dataset = get_dataset('mdoc_eval', "../datasets/hotpotqa_train.jsonl", **dataset_args)
        # musique_train = get_dataset('mdoc', "../datasets/musique_ans_v1.0_train.jsonl", **dataset_args)
        # musique_eval = get_dataset('mdoc_eval', "../datasets/musique_ans_v1.0_train.jsonl", **dataset_args)
        train_dataset.data = train_dataset.data.select(range(60000))
        # load longmagpie QA dataset
        longmagpie_path = os.path.join(training_args.train_data, "longmagpie_processed")
        longmagpie_train = get_dataset('mdoc', longmagpie_path, **dataset_args)
        # longmagpie_eval = get_dataset('mdoc_eval', longmagpie_path, **dataset_args)
        longmagpie_train.data = longmagpie_train.data.select(range(60000))
        # dataset_args.pop('max_doc_length')
        # dataset_args.pop('max_doc_num')
        # load slimpajamas pretrain dataset
        # dataset_args.update({
            # 'max_length': train_dataset.max_length + train_dataset.max_doc_length * train_dataset.max_doc_num, 
            # 'min_length': train_dataset.max_doc_length * 2,
            # 'num_samples': 10000, 'streaming': False, 'cut_long_seq': True,
        # })
        # slimpajamas_path = os.path.join(training_args.train_data, "slimpajamas_subset")
        # slimpajamas_train = get_dataset('pretrain', slimpajamas_path, **dataset_args).to_mdoc_format2(tokenizer, train_dataset)
        # print the lengths of all training datasets
        # logger.info(f"Train dataset lengths: musique: {len(musique_train)}, longmagpie: {len(longmagpie_train)}, hotpotqa: {len(train_dataset)}")
        train_dataset.merge([longmagpie_train])
        # eval_dataset.merge([musique_eval], method='concat')

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
