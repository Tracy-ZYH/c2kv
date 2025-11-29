import torch
from packaging import version
from logging import getLogger
from typing import Tuple, Type, Optional
from dataclasses import asdict
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers.utils import logging
from transformers.integrations import is_deepspeed_zero3_enabled

from gist_args import ModelArgs

logger = getLogger(__name__)


class GistLossFunctionWithRegularization:
    def __init__(self, model: PreTrainedModel, regularization_strength: float):
        super().__init__()
        self.model = model
        self.label_loss_function = model.loss_function
        self.regularization_strength = regularization_strength
    
    @staticmethod
    def _compute_weight_loss(gist_weight: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (gist_weight - weight).pow(2).mean()
    
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        num_items_in_batch: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        label_loss = self.label_loss_function(logits, labels, vocab_size, num_items_in_batch, ignore_index, shift_labels, **kwargs)
        regularization_losses = []
        for layer in self.model.model.layers:
            attn = layer.self_attn
            regularization_losses.extend([
                self._compute_weight_loss(attn.gist_q_proj.weight, attn.q_proj.weight),
                self._compute_weight_loss(attn.gist_k_proj.weight, attn.k_proj.weight),
                self._compute_weight_loss(attn.gist_v_proj.weight, attn.v_proj.weight),
            ])
        loss = label_loss + torch.stack(regularization_losses).mean() * self.regularization_strength
        return loss


def get_model_class(model_name_or_path: str) -> Tuple[Type[PretrainedConfig], Type[PreTrainedModel]]:
    from .llama import LlamaForCausalLM, LlamaConfig
    from .qwen3 import Qwen3ForCausalLM, Qwen3Config
    ARCHITECTURE_TO_CLASS = {
        'LlamaForCausalLM': (LlamaConfig, LlamaForCausalLM),
        'Qwen3ForCausalLM': (Qwen3Config, Qwen3ForCausalLM),
    }
    probe_config = AutoConfig.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True,
        local_files_only=True,
    )
    architecture = probe_config.architectures[0]
    config_class, model_class = ARCHITECTURE_TO_CLASS[architecture]
    return config_class, model_class


def get_model_and_tokenizer(
    model_args: ModelArgs, 
    device: str="cuda", 
    evaluation_mode: bool=True, 
    return_tokenizer_only: bool=False, 
    **kwargs
):    
    model_args_dict = asdict(model_args)
    model_args_dict.update(**kwargs)
    
    model_name_or_path = model_args_dict["model_name_or_path"]
    cache_dir = model_args_dict["model_cache_dir"]
    # access_token = model_args_dict["access_token"]

    logger.info(f"Loading model and tokenizer from {model_name_or_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        cache_dir=cache_dir, 
        padding_side=model_args_dict["padding_side"], 
        # token=access_token, 
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if return_tokenizer_only:
        return tokenizer

    dtype = model_args_dict["dtype"]
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    device_map = model_args_dict["device_map"]
    if device_map is None and not is_deepspeed_zero3_enabled():
        device_map = {"": device}
    
    rope_kwargs = {}
    rope_theta = model_args_dict["rope_theta"]
    if rope_theta is not None:
        rope_kwargs["rope_theta"] = rope_theta
    rope_method = model_args_dict["rope_method"]
    if rope_method is not None:
        rope_factor = model_args_dict["rope_factor"]
        rope_scaling = {
            "type": rope_method,
            "factor": rope_factor
        }
        # NOTE: do not destroy the default rope_scaling of the model
        rope_kwargs["rope_scaling"] = rope_scaling

    attn_kwargs = {}
    attn_impl = model_args_dict["attn_impl"]
    if attn_impl is not None:
        if version.parse(transformers.__version__) <= version.parse("4.36"):
            if attn_impl == "flash_attention_2":
                attn_kwargs["use_flash_attention_2"] = True
        else:
            attn_kwargs["attn_implementation"] = attn_impl

    gist_kwargs = {}
    for k, v in model_args_dict.items():
        if k.startswith("gist") and v is not None:
            gist_kwargs[k] = v
    
    # use architecture attribute to distinguish different models
    probe_config = AutoConfig.from_pretrained(
        model_name_or_path, 
        cache_dir=cache_dir, 
        # token=access_token, 
        trust_remote_code=True,
        local_files_only=True,
    )
    architecture = probe_config.architectures[0]

    if model_args_dict["enable_gist"]:
        config_class, model_class = get_model_class(model_name_or_path)

        config = config_class.from_pretrained(
            model_name_or_path, 
            # cache_dir=cache_dir,
            # token=access_token,
            # NOTE: keep the torch_dtype in config consistent with that in model
            dtype=dtype,
            local_files_only=True,
            **gist_kwargs,
            **rope_kwargs,
            **attn_kwargs,
        )
        config.gist_token_id = tokenizer.eos_token_id
        config.gist_type = model_args_dict["gist_type"]
        model = model_class.from_pretrained(
            model_name_or_path, 
            config=config,
            # cache_dir=cache_dir, 
            dtype=dtype,
            device_map=device_map, 
            # token=access_token,
            local_files_only=True,
        )

        if model_args_dict["gist_regularization"] is not None:
            model.loss_function = GistLossFunctionWithRegularization(
                model, model_args_dict["gist_regularization"]
            )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir, 
            dtype=dtype,
            device_map=device_map,
            # token=access_token,
            trust_remote_code=True,
            # NOTE: do not destroy the default rope_scaling of the model
            **rope_kwargs,
            **attn_kwargs,
        )

    # load lora
    if model_args_dict["lora"] is not None:
        logger.info(f"loading lora from {model_args_dict['lora']}...")

        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, 
            model_args_dict["lora"],
            torch_dtype=dtype,
            device_map=device_map,
        )
        if model_args_dict["lora_unload"]:
            model = model.merge_and_unload()

    if model_args_dict["enable_tp"]:
        import tensor_parallel as tp
        logger.info("enabling tensor parallelism...")
        
        # model = tp.tensor_parallel(model, device_ids=list(range(8)), distributed=False, sharded=False)
        model = tp.tensor_parallel(model, sharded=True)

        if model.generation_config.eos_token_id == 128001:
            model.generation_config.eos_token_id = [128001, 128009]

    if isinstance(model, transformers.modeling_utils.PreTrainedModel):
        model = model.eval()
        if evaluation_mode:
            # NOTE: essential to disable all gradient in-place, so that when calling accelerator.prepare, the forward function will not be wrapped that may consume extra GPU memory
            model.requires_grad_(False)
        # logger.info(model.config)

    # override the default generation config
    generation_config = model_args.get_generation_config()
    if len(generation_config):
        model.generation_config.update(**generation_config)
    logger.info(f"Specified generation config: {generation_config}")

    return model, tokenizer

def format_numel_str(numel: int) -> str:
    T = 1e12
    B = 1e9
    M = 1e6
    K = 1e3
    if numel >= T:
        return f"{numel / T:.2f} T"
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"