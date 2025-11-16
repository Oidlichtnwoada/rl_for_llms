import typing
from copy import deepcopy

from datasets import Dataset, Value, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.constant_utils import (
    get_default_evaluation_file_names,
    get_gitignore_file_name,
    get_hf_training_ds_path,
    get_relative_training_file_path,
    get_train_split,
)
from rl_for_llms.utils.dataset_utils import trim_dataset
from rl_for_llms.utils.llm_utils import get_user_message
from rl_for_llms.utils.path_utils import (
    get_checkpoint_folder_for_model_id,
    get_evaluation_data_dir,
    get_training_data_dir,
    is_folder_empty,
)
from rl_for_llms.utils.reward_utils import default_batch_reward_function
from rl_for_llms.utils.torch_utils import get_cuda_device_count, is_cuda_device_used


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
    dataset = dataset.remove_columns(["ability", "data_source"])
    ids = [str(x["index"]) for x in list(dataset["extra_info"])]
    dataset = dataset.add_column("id", ids)
    dataset = dataset.remove_columns(["extra_info"])
    answers = [str(x["ground_truth"]) for x in list(dataset["reward_model"])]
    dataset = dataset.add_column("answer", answers)
    dataset = dataset.remove_columns(["reward_model"])
    return dataset


def load_evaluation_data() -> Dataset:
    """Load evaluation data."""
    file_paths = [
        get_evaluation_data_dir() / file_name
        for file_name in get_default_evaluation_file_names()
    ]
    dataset_dicts = [
        load_dataset("json", data_files=[str(file_path.resolve())])
        for file_path in file_paths
    ]
    datasets = [dataset_dict[str(get_train_split())] for dataset_dict in dataset_dicts]
    target_columns = {"problem", "answer", "id"}
    for index, dataset in enumerate(datasets):
        if "unique_id" in dataset.column_names:
            new_dataset = dataset.rename_columns({"unique_id": "id"})
        else:
            new_dataset = deepcopy(dataset)
        columns_to_drop = set(new_dataset.column_names) - target_columns
        new_dataset = new_dataset.remove_columns(list(columns_to_drop))
        for column in target_columns:
            new_dataset = new_dataset.cast_column(column, Value("string"))
        datasets[index] = new_dataset
    merged_dataset = concatenate_datasets(datasets)
    prompts = [[get_user_message(x)] for x in list(merged_dataset["problem"])]
    merged_dataset = merged_dataset.add_column("prompt", prompts)
    merged_dataset = merged_dataset.remove_columns(["problem"])
    return merged_dataset


def get_grpo_config() -> GRPOConfig:
    """Get the GRPO configuration."""
    config = get_config()
    vllm_tensor_parallel_size = get_cuda_device_count() if is_cuda_device_used() else 1
    grpo_config = GRPOConfig(
        max_prompt_length=config.max_prompt_length,
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


def get_tokenizer(
    model_id: str, truncation_side: str = "left"
) -> PreTrainedTokenizerBase:
    """Return the tokenizer for the specified model ID."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, truncation_side=truncation_side, trust_remote_code=True
    )  # type: ignore[no-untyped-call]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return typing.cast("PreTrainedTokenizerBase", tokenizer)


def get_grpo_trainer() -> GRPOTrainer:
    """Get the GRPO trainer."""
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
    grpo_trainer = GRPOTrainer(
        model=config.hf_model_id,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=[default_batch_reward_function],
        peft_config=peft_config if config.enable_lora else None,
    )
    return grpo_trainer


def start_training() -> None:
    """Start the training process."""
    download_training_data()
    grpo_trainer = get_grpo_trainer()
    grpo_trainer.train()
