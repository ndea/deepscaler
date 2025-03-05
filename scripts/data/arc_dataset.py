"""Script to prepare ARG-AGI dataset for DeepScaler training.

This script loads the ARG-AGI dataset from Hugging Face and processes it
into a standardized format for training DeepScaler models.
"""

import argparse
import json
import os
from typing import Any, Dict, Set

import pandas as pd
from datasets import Dataset, load_dataset

from verl.utils.hdfs_io import copy, makedirs


def filter_dataset(dataset: Dataset) -> Dataset:
    """
    Filter the dataset to only include challenges with correct IDs.

    Args:
        dataset: The original dataset

    Returns:
        The filtered dataset
    """
    try:
        # Try to load correct_challenge_ids.json, if available
        correct_ids_path = os.path.join(
            os.path.dirname(__file__), "correct_challenge_ids.json"
        )

        if os.path.exists(correct_ids_path):
            ids_to_keep: Set[str] = {
                i["challenge_id"] for i in json.loads(open(correct_ids_path).read())
            }

            new_ids_to_keep: Set[str] = set()
            for i in ids_to_keep:
                if "_" not in i:
                    i = f"{i}-0"
                else:
                    i = i.replace("_", "-")
                new_ids_to_keep.add(i)

            print(
                f"Filtering dataset to {len(new_ids_to_keep)} challenges with correct IDs"
            )
            return dataset.filter(lambda x: x["id"] in new_ids_to_keep)
        else:
            print("correct_challenge_ids.json not found, using full dataset")
            return dataset
    except Exception as e:
        print(f"Error filtering dataset: {e}")
        print("Using full dataset instead")
        return dataset


def make_conversation(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts an example into a conversation with a system prompt.
    The system prompt instructs the assistant to provide reasoning within <think>...</think>
    and the final answer in <answer>...</answer>.

    Args:
        example: The dataset example

    Returns:
        A dictionary with the formatted conversation
    """
    system_prompt = (
        "A conversation between User and Assistant. The assistant first shows its reasoning inside <think>...</think> "
        "and then provides the final answer inside <answer>...</answer>."
    )

    # Extract problem statement
    problem = example.get("problem", "")

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]
    }


def process_arc_agi_dataset(local_dir, hdfs_dir=None):
    """
    Process the ARG-AGI dataset into the DeepScaler format.

    Args:
        local_dir: Directory to save the processed dataset
        hdfs_dir: Optional HDFS directory to copy the dataset to

    Returns:
        Path to the saved parquet file
    """
    print("Loading ARG-AGI dataset from Hugging Face...")
    # Load the dataset from Hugging Face
    dataset = load_dataset("jerber/arc-agi")

    # Make local directory if it doesn't exist
    makedirs(local_dir)

    # Check which split is available (usually 'train')
    available_splits = dataset.keys()
    primary_split = (
        "train" if "train" in available_splits else list(available_splits)[0]
    )

    # Filter the dataset if possible
    dataset_split = dataset[primary_split]
    filtered_dataset = filter_dataset(dataset_split)

    print(
        f"Processing {len(filtered_dataset)} examples from the {primary_split} split..."
    )

    # Process the dataset into the expected format
    train_data = []
    for idx, item in enumerate(filtered_dataset):
        # Create a unique ID for the example
        example_id = item.get("id", f"example-{idx}")

        # Get problem and solution
        problem = item.get("problem", "")
        solution = item.get("solution", "")

        # Format as a conversation with the special format
        conversation = make_conversation(item)

        # Create the data format expected by DeepScaler
        data = {
            "data_source": "arc-agi",
            "prompt": conversation["prompt"],
            "ability": "reasoning",
            "reward_model": {
                "style": "arc",
                "ground_truth": {"solution": solution, "id": example_id},
                "module_path": "deepscaler.data.arc_reward_model",
                "class_name": "ARCRewardModel",
            },
            "extra_info": {"split": "train", "index": idx, "id": example_id},
        }
        train_data.append(data)

    # Save training dataset
    train_file = os.path.join(local_dir, "arc_agi_train.parquet")
    print(f"Saving {len(train_data)} training examples to {train_file}")
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(train_file)

    # Create test dataset (using a subset of the data)
    # For simplicity, we'll use the last 10% as a validation set
    test_size = max(1, len(train_data) // 10)
    test_data = train_data[-test_size:]
    for item in test_data:
        item["extra_info"]["split"] = "test"

    # Save test dataset
    test_file = os.path.join(local_dir, "arc_agi_test.parquet")
    print(f"Saving {len(test_data)} test examples to {test_file}")
    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(test_file)

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=train_file, dst=os.path.join(hdfs_dir, "arc_agi_train.parquet"))
        copy(src=test_file, dst=os.path.join(hdfs_dir, "arc_agi_test.parquet"))
        print(f"Copied files to HDFS: {hdfs_dir}")

    return train_file, test_file


if __name__ == "__main__":
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
        "--correct_ids_path",
        default=None,
        help="Path to correct_challenge_ids.json for filtering (optional)",
    )
    args = parser.parse_args()

    # If correct_ids_path is provided, copy it to the script directory
    if args.correct_ids_path and os.path.exists(args.correct_ids_path):
        script_dir = os.path.dirname(__file__)
        dest_path = os.path.join(script_dir, "correct_challenge_ids.json")
        try:
            import shutil

            shutil.copy(args.correct_ids_path, dest_path)
            print(
                f"Copied correct_challenge_ids.json from {args.correct_ids_path} to {dest_path}"
            )
        except Exception as e:
            print(f"Error copying correct_challenge_ids.json: {e}")

    train_file, test_file = process_arc_agi_dataset(args.local_dir, args.hdfs_dir)
    print(
        f"ARG-AGI dataset processing complete. Files saved to:\n- {train_file}\n- {test_file}"
    )
