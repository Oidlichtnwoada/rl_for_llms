import typing

from datasets import Dataset, load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer, ProcessorMixin
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.constant_utils import (
    get_default_evaluation_file_name,
    get_gitignore_file_name,
    get_hf_training_ds_path,
    get_relative_training_file_path,
    get_train_split,
)
from rl_for_llms.utils.dataset_utils import trim_dataset
from rl_for_llms.utils.path_utils import (
    get_checkpoint_folder_for_model_id,
    get_evaluation_data_dir,
    get_training_data_dir,
    is_folder_empty,
)
from rl_for_llms.utils.reward_utils import default_reward_function


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


def load_training_data_from_disk() -> Dataset:
    """Load training data previously saved to disk."""
    training_data_dir = get_training_data_dir()
    dataset = load_from_disk(training_data_dir)
    return dataset


def load_evaluation_data() -> Dataset:
    """Load evaluation data."""
    file_path = get_evaluation_data_dir() / get_default_evaluation_file_name()
    dataset_dict = load_dataset("json", data_files=[str(file_path.resolve())])
    dataset = dataset_dict[str(get_train_split())]
    return dataset


def get_grpo_config() -> GRPOConfig:
    """Get the GRPO configuration."""
    config = get_config()
    grpo_config = GRPOConfig(
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        num_generations=config.num_generations,
        temperature=config.temperature,
        top_p=config.top_p,
        use_vllm=config.use_vllm,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
        vllm_mode=config.vllm_mode,
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
    )
    return grpo_config


def get_tokenizer(model_id: str, truncation_side: str = "left") -> ProcessorMixin:
    """Return the tokenizer for the specified model ID."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, truncation_side=truncation_side, trust_remote_code=True
    )  # type: ignore[no-untyped-call]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return typing.cast("ProcessorMixin", tokenizer)


def get_grpo_trainer() -> GRPOTrainer:
    """Get the GRPO trainer."""
    config = get_config()
    grpo_config = get_grpo_config()
    train_dataset = trim_dataset(
        load_training_data_from_disk(), config.dataset_use_row_percentage
    )
    eval_dataset = trim_dataset(
        load_evaluation_data(), config.dataset_use_row_percentage
    )
    tokenizer = get_tokenizer(config.hf_model_id)
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha_factor_for_rank * config.lora_rank,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type,
    )
    grpo_trainer = GRPOTrainer(
        model=config.hf_model_id,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=[default_reward_function],
        peft_config=peft_config,
    )
    return grpo_trainer


def start_training() -> None:
    """Start the training process."""
    download_training_data()
    grpo_trainer = get_grpo_trainer()
    grpo_trainer.train()
