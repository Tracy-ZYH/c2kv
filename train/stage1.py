import logging
from transformers import HfArgumentParser, DataCollatorWithPadding
from transformers.integrations import is_deepspeed_zero3_enabled

from .train_data import get_dataset
from .trainer import GistTrainer
from models.model_utils import get_model_and_tokenizer, format_numel_str
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

    model, tokenizer = get_model_and_tokenizer(model_args, evaluation_mode=False)

    if model_args.enable_gist and training_args.only_train_gist:
        for name, param in model.named_parameters():
            param.requires_grad_('gist' in name)
    
    logger.info(f"Total Model params: {format_numel_str(sum(p.numel() for p in model.parameters()))}")
    logger.info(f"Trainable Model params: {format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}")

    with training_args.main_process_first():
        train_dataset = get_dataset(
            'pretrain', training_args.train_data, tokenizer, 
            max_length=training_args.pretrain_max_length,
            min_length=training_args.pretrain_min_length,
        )

    trainer = GistTrainer(
        model=model,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt'),
    )

    if train_dataset is not None:
        trainer.train()
    
if __name__ == "__main__":
    main()
