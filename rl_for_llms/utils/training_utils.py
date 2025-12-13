from datasets import load_dataset
from peft import LoraConfig
from trl.trainer.grpo_config import GRPOConfig

from rl_for_llms.utils.confidence_grpo_trainer_utils import ConfidenceGRPOTrainer
from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.constant_utils import (
    get_gitignore_file_name,
    get_hf_training_ds_path,
    get_relative_training_file_path,
    get_train_split,
)
from rl_for_llms.utils.dataset_utils import (
    load_evaluation_data,
    load_training_data_from_disk,
    trim_dataset,
)
from rl_for_llms.utils.llm_utils import get_tokenizer
from rl_for_llms.utils.logging_utils import log_msg
from rl_for_llms.utils.path_utils import (
    get_checkpoint_folder_for_model_id,
    get_training_data_dir,
    is_folder_empty,
)
from rl_for_llms.utils.reward_utils import default_batch_reward_function
from rl_for_llms.utils.torch_utils import (
    get_cuda_device_count,
    is_cuda_device_used,
    setup_environment,
)


def download_training_data() -> None:
    """Download training data from a remote source."""
    training_data_dir = get_training_data_dir()
    if not is_folder_empty(
        training_data_dir, ignore_file_names=(get_gitignore_file_name(),)
    ):
        return
    dataset = load_dataset(
        path=get_hf_training_ds_path(),
        data_files=get_relative_training_file_path(),
        split=get_train_split(),
    )
    dataset.save_to_disk(training_data_dir)


def get_grpo_config() -> GRPOConfig:
    """Get the GRPO configuration."""
    config = get_config()
    vllm_tensor_parallel_size = get_cuda_device_count() if is_cuda_device_used() else 1
    grpo_config = GRPOConfig(
        max_completion_length=config.max_completion_length,
        num_generations=config.num_generations,
        temperature=config.temperature,
        top_p=config.top_p,
        use_vllm=config.use_vllm,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size
        if config.vllm_split_model_across_gpus
        else 1,
        vllm_mode=config.vllm_mode,
        vllm_enable_sleep_mode=config.vllm_enable_sleep_mode,
        vllm_importance_sampling_correction=config.vllm_importance_sampling_correction,
        learning_rate=config.learning_rate,
        output_dir=str(
            get_checkpoint_folder_for_model_id(config.hf_model_id).resolve()
        ),
        num_train_epochs=config.num_train_epochs,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        per_device_train_batch_size=config.per_device_rollouts_per_batch
        * config.num_generations,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=config.report_to,
        log_completions=config.log_completions,
        run_name=config.hf_model_id,
        beta=config.beta,
        dataloader_pin_memory=config.dataloader_pin_memory,
    )
    return grpo_config


def get_confidence_grpo_trainer() -> ConfidenceGRPOTrainer:
    """Get the confidence GRPO trainer."""
    config = get_config()
    grpo_config = get_grpo_config()
    tokenizer = get_tokenizer(config.hf_model_id)
    train_dataset = trim_dataset(
        load_training_data_from_disk(),
        config.dataset_use_row_percentage,
        tokenizer,
        config.max_prompt_length,
    )
    eval_dataset = trim_dataset(
        load_evaluation_data(),
        config.dataset_use_row_percentage,
        tokenizer,
        config.max_prompt_length,
    )
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha_factor_for_rank * config.lora_rank,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=config.lora_task_type,
    )
    confidence_grpo_trainer = ConfidenceGRPOTrainer(
        config=config,
        model=config.hf_model_id,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=[default_batch_reward_function],
        peft_config=peft_config if config.enable_lora else None,
    )
    return confidence_grpo_trainer


def start_training() -> None:
    """Start the training process."""
    setup_environment()
    config = get_config()
    download_training_data()
    confidence_grpo_trainer = get_confidence_grpo_trainer()
    log_msg(
        f"start training with the following configuration: {config.model_dump_json()}"
    )
    confidence_grpo_trainer.train()
    log_msg(
        f"finished training with the following configuration: {config.model_dump_json()}"
    )
