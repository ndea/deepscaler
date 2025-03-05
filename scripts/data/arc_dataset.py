"""
Script to prepare ARG-AGI dataset for DeepScaler training.

This script loads the ARG-AGI dataset from a file (or Hugging Face) and processes it
into a standardized format for training DeepScaler models.
"""

import argparse
import json
import os
import shutil
from typing import Any, Dict, Set, Tuple

import pandas as pd
from datasets import Dataset, load_dataset

from verl.utils.hdfs_io import copy, makedirs


def filter_dataset(dataset: Dataset) -> Dataset:
    """
    Filter the training dataset to only include challenges with correct IDs.
    Assumes that correct_challenge_ids.json always exists in the same directory as the script.
    
    Args:
        dataset: The original training dataset.
        
    Returns:
        The filtered dataset.
    """
    correct_ids_path = os.path.join(os.path.dirname(__file__), "correct_challenge_ids.json")
    print(f"Loading correct IDs from {correct_ids_path}")
    with open(correct_ids_path, "r") as f:
        ids_data = json.load(f)

    # Create a set of formatted challenge IDs
    ids_to_keep: Set[str] = set()
    for entry in ids_data:
        challenge_id = entry.get("challenge_id", "")
        formatted_id = challenge_id.replace("_", "-") if "_" in challenge_id else f"{challenge_id}-0"
        ids_to_keep.add(formatted_id)

    print(f"Filtering dataset from {len(dataset)} examples to keep {len(ids_to_keep)} challenge IDs")
    filtered = dataset.filter(lambda x: x.get("id", "") in ids_to_keep)
    print(f"After filtering: {len(filtered)} examples remain")
    if len(filtered) == 0:
        print("WARNING: Filtering resulted in empty dataset, using full dataset instead")
        return dataset
    return filtered


def make_conversation(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an example into a conversation with a system prompt.
    The system prompt instructs the assistant to provide reasoning within <think>...</think>
    and the final answer in <answer>...</answer>.

    Args:
        example: The dataset example.

    Returns:
        A dictionary with the formatted conversation.
    """
    system_prompt = (
        "A conversation between User and Assistant. The assistant first shows its reasoning inside <think>...</think> "
        "and then provides the final answer inside <answer>...</answer>."
    )
    problem = example.get("problem", "")
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]
    }


def process_split(dataset: Dataset, split: str) -> list:
    """
    Process a dataset split into the format expected by DeepScaler.

    Args:
        dataset: The dataset split.
        split: The split name ('train' or 'test').

    Returns:
        A list of processed examples.
    """
    processed_data = []
    for idx, item in enumerate(dataset):
        if "prompt" not in item:
            item.update(make_conversation(item))
        example_id = item.get("id", f"{split}-{idx}")
        solution = item.get("solution", "")
        processed_example = {
            "data_source": "arc-agi",
            "prompt": item["prompt"],
            "ability": "reasoning",
            "reward_model": {
                "style": "arc",
                "ground_truth": {"solution": solution, "id": example_id},
                "module_path": "deepscaler.data.arc_reward_model",
                "class_name": "ARCRewardModel",
            },
            "extra_info": {"split": split, "index": idx, "id": example_id},
        }
        processed_data.append(processed_example)
    return processed_data


def process_dataset(train_dataset: Dataset, test_dataset: Dataset, local_dir: str, hdfs_dir: str = None) -> Tuple[str, str]:
    """
    Process datasets into the DeepScaler format and save them as parquet files.

    Args:
        train_dataset: The training dataset.
        test_dataset: The test dataset.
        local_dir: Directory to save the processed dataset.
        hdfs_dir: Optional HDFS directory to copy the dataset to.

    Returns:
        Paths to the saved parquet files.
    """
    # Ensure local directory exists; if it already exists, skip creation.
    if not os.path.exists(local_dir):
        makedirs(local_dir)
    else:
        print(f"Local directory {local_dir} already exists, skipping makedirs.")

    # Process and save training data
    print(f"Processing {len(train_dataset)} training examples...")
    train_data = process_split(train_dataset, "train")
    train_file = os.path.join(local_dir, "arc_agi_train.parquet")
    print(f"Saving {len(train_data)} training examples to {train_file}")
    pd.DataFrame(train_data).to_parquet(train_file)

    # Process and save test data
    print(f"Processing {len(test_dataset)} test examples...")
    test_data = process_split(test_dataset, "test")
    test_file = os.path.join(local_dir, "arc_agi_test.parquet")
    print(f"Saving {len(test_data)} test examples to {test_file}")
    pd.DataFrame(test_data).to_parquet(test_file)

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        if not os.path.exists(hdfs_dir):
            makedirs(hdfs_dir)
        else:
            print(f"HDFS directory {hdfs_dir} already exists, skipping makedirs.")
        copy(src=train_file, dst=os.path.join(hdfs_dir, "arc_agi_train.parquet"))
        copy(src=test_file, dst=os.path.join(hdfs_dir, "arc_agi_test.parquet"))
        print(f"Copied files to HDFS: {hdfs_dir}")

    return train_file, test_file


def load_from_huggingface() -> Tuple[Dataset, Dataset]:
    """Load the ARG-AGI dataset from Hugging Face."""
    print("Loading ARG-AGI dataset from Hugging Face...")
    dataset = load_dataset("jerber/arc-agi")
    available_splits = dataset.keys()
    primary_split = "train" if "train" in available_splits else list(available_splits)[0]

    if "test" not in available_splits:
        print("No test split found, creating a 90/10 train-test split...")
        split_data = dataset[primary_split].train_test_split(test_size=0.1)
        return split_data["train"], split_data["test"]
    return dataset["train"], dataset["test"]


def load_local_dataset(dataset_path: str) -> Tuple[Dataset, Dataset]:
    """
    Load dataset from a local file.

    Args:
        dataset_path: Path to the dataset file.

    Returns:
        A tuple of training and test datasets.
    """
    print(f"Loading dataset from {dataset_path}...")
    if dataset_path.endswith(".json"):
        with open(dataset_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and {"train", "test"}.issubset(data):
            train_data, test_data = data["train"], data["test"]
        else:
            split_point = int(len(data) * 0.9)
            train_data, test_data = data[:split_point], data[split_point:]
        return Dataset.from_list(train_data), Dataset.from_list(test_data)
    else:
        try:
            return load_dataset(dataset_path, split=["train", "test"])
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to Hugging Face dataset")
            return load_from_huggingface()


def main():
    parser = argparse.ArgumentParser(
        description="Process ARG-AGI dataset for DeepScaler training"
    )
    parser.add_argument(
        "--local_dir",
        default=os.path.expanduser("~/deepscaler/data"),
        help="Local directory to save processed datasets",
    )
    parser.add_argument(
        "--hdfs_dir", default=None, help="Optional HDFS directory to copy datasets to"
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Path to dataset file (if not using Hugging Face)",
    )
    args = parser.parse_args()

    # Load dataset from local file or Hugging Face
    if args.dataset_path:
        train_dataset, test_dataset = load_local_dataset(args.dataset_path)
    else:
        train_dataset, test_dataset = load_from_huggingface()

    # Filter the training dataset (apply only on train data)
    print("Filtering training dataset...")
    train_dataset = filter_dataset(train_dataset)

    # Apply conversation format to both datasets
    print("Applying conversation format...")
    train_dataset = train_dataset.map(make_conversation)
    test_dataset = test_dataset.map(make_conversation)

    # Remove unnecessary columns if they exist
    columns_to_remove = ["id", "problem"]
    print(f"Removing unnecessary columns: {columns_to_remove}")
    for col in columns_to_remove:
        if col in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns([col])
        if col in test_dataset.column_names:
            test_dataset = test_dataset.remove_columns([col])

    # Process and save the datasets
    train_file, test_file = process_dataset(train_dataset, test_dataset, args.local_dir, args.hdfs_dir)
    print(
        f"ARG-AGI dataset processing complete. Files saved to:\n- {train_file}\n- {test_file}"
    )


if __name__ == "__main__":
    main()