import typing

from datasets import Dataset, load_dataset, load_from_disk
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
    dataset = load_dataset("json", data_files=[str(file_path.resolve())])
    return dataset


def get_grpo_config() -> GRPOConfig:
    """Get the GRPO configuration."""
    config = get_config()
    grpo_config = GRPOConfig(
        max_prompt_length=2048,
        max_completion_length=8192,
        num_generations=8,
        temperature=1.0,
        top_p=1.0,
        use_vllm=config.use_vllm,
        vllm_gpu_memory_utilization=0.6,
        vllm_tensor_parallel_size=1,
        vllm_mode="colocate",
        learning_rate=config.learning_rate,
        output_dir=str(
            get_checkpoint_folder_for_model_id(config.hf_model_id).resolve()
        ),
        num_train_epochs=config.num_train_epochs,
        save_steps=50,
        eval_steps=20,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        report_to=config.report_to,
        log_completions=True,
        run_name=config.hf_model_id,
        beta=0.0,
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
    train_dataset = load_training_data_from_disk()
    eval_dataset = load_evaluation_data()
    tokenizer = get_tokenizer(config.hf_model_id)
    grpo_trainer = GRPOTrainer(
        model=config.hf_model_id,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=[default_reward_function],
        peft_config=None,
    )
    return grpo_trainer


def start_training() -> None:
    """Start the training process."""
    download_training_data()
    grpo_trainer = get_grpo_trainer()
    grpo_trainer.train()
