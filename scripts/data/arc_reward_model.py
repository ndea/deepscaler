"""
Custom reward model for ARG-AGI dataset.

This module implements a reward model for the ARG-AGI dataset that extracts
answers from model responses and computes rewards based on comparison with
ground truth.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from verl.utils.reward_score.base import BaseRewardModel


class ARCRewardModel(BaseRewardModel):
    """Custom reward model for ARG-AGI dataset."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def format_reward(output: str) -> float:
        """
        Returns a reward of 1.0 if the generated completion strictly follows the format:
        a reasoning enclosed in <think>...</think> and a final answer in <answer>...</answer>.

        Args:
            output: The model's raw response

        Returns:
            1.0 if the format is correct, 0.0 otherwise
        """
        pattern = r"^<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>$"
        return 1.0 if re.match(pattern, output.strip()) else 0.0

    @staticmethod
    def extract_arc_answer(text: str) -> str:
        """Extract the answer from a model response.

        Tries to extract content from <answer>...</answer> tags.

        Args:
            text: The raw model response text

        Returns:
            The extracted answer as a string
        """
        # Look for content within <answer> tags
        answer_pattern = r"<answer>([\s\S]*?)</answer>"
        answer_match = re.search(answer_pattern, text)

        if answer_match:
            return answer_match.group(1).strip()

        # If no answer tag is found, try other methods

        # First try to extract from boxed content
        boxed_pattern = r"\\boxed{([^}]*)}"
        boxed_matches = re.findall(boxed_pattern, text)
        if boxed_matches:
            return boxed_matches[-1].strip()  # Return the last boxed answer

        # Try to find lines starting with "Answer:" or similar patterns
        answer_patterns = [
            r"(?:^|\n)Answer:\s*(.*?)(?:\n|$)",
            r"(?:^|\n)The answer is:\s*(.*?)(?:\n|$)",
            r"(?:^|\n)The final answer is:\s*(.*?)(?:\n|$)",
            r"(?:^|\n)Therefore,\s*(.*?)(?:\n|$)",
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].strip()

        # As a last resort, return the last non-empty line
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            return lines[-1]

        # If we still haven't found anything, return the original text
        return text.strip()

    @staticmethod
    def grid_from_str(s: str) -> list[list[int]]:
        """
        Extract a grid (2D list of integers) from a string.

        Args:
            s: String possibly containing a grid

        Returns:
            A 2D list of integers representing the grid
        """

        def is_int(token: str) -> bool:
            try:
                int(token)
                return True
            except ValueError:
                return False

        lines = s.splitlines()
        final_grid = []
        current_grid = []

        for line in lines:
            tokens = line.split()
            if tokens and all(is_int(token) for token in tokens):
                current_grid.append([int(token) for token in tokens])
            else:
                if current_grid:
                    final_grid = current_grid
                    current_grid = []

        if current_grid:
            final_grid = current_grid

        return final_grid

    @staticmethod
    def is_valid_2d_int_grid(grid: Any) -> bool:
        """
        Check if grid is a valid 2D list of integers.

        Args:
            grid: Object to check

        Returns:
            True if grid is a valid 2D list of integers, False otherwise
        """
        if not isinstance(grid, list) or not grid:
            return False
        if not all(isinstance(row, list) for row in grid):
            return False
        if not all(isinstance(num, int) for row in grid for num in row):
            return False
        if not all(len(row) == len(grid[0]) for row in grid):
            return False
        return True

    def accuracy_reward(self, output: str, solution_str: str) -> float:
        """
        Calculate accuracy reward based on comparing grids.

        Args:
            output: The model's raw response
            solution_str: The ground truth solution as a string

        Returns:
            A reward value between 0.0 and 10.0
        """
        try:
            answer_grid = self.grid_from_str(output)
            solution_grid = json.loads(solution_str)

            reward = 0.0

            if not self.is_valid_2d_int_grid(
                answer_grid
            ) or not self.is_valid_2d_int_grid(solution_grid):
                return reward

            if len(answer_grid) == len(solution_grid):
                reward += 2
            if len(answer_grid[0]) == len(solution_grid[0]):
                reward += 2

            if answer_grid == solution_grid:
                reward += 6

            return reward

        except (json.JSONDecodeError, IndexError, TypeError, AttributeError) as e:
            print(f"Error in accuracy reward: {e}")
            return 0.0

    def compute_reward_score(
        self,
        output: str,
        ground_truth: Optional[Union[str, List, Dict]] = None,
        prompt: Optional[Dict] = None,
    ) -> Tuple[float, Dict]:
        """
        Compute the reward score for the ARG-AGI dataset.

        Args:
            output: The model's raw response
            ground_truth: The expected answer
            prompt: The original prompt (unused)

        Returns:
            Tuple of (score, metadata)
        """
        if not ground_truth:
            return 0.0, {"error": "No ground truth provided"}

        # First check format compliance
        format_score = self.format_reward(output)

        # Extract the answer from the model output
        extracted_answer = self.extract_arc_answer(output)

        # Compute accuracy reward if possible
        accuracy_score = 0.0
        try:
            if isinstance(ground_truth, str):
                accuracy_score = self.accuracy_reward(output, ground_truth)
            elif isinstance(ground_truth, dict) and "solution" in ground_truth:
                accuracy_score = self.accuracy_reward(output, ground_truth["solution"])
        except Exception as e:
            print(f"Error computing accuracy score: {e}")

        # Convert accuracy score (0-10) to 0-1 range
        normalized_accuracy = min(accuracy_score / 10.0, 1.0)

        # Final score combines format and accuracy weights
        FORMAT_WEIGHT = 0.3
        ACCURACY_WEIGHT = 0.7

        final_score = (FORMAT_WEIGHT * format_score) + (
            ACCURACY_WEIGHT * normalized_accuracy
        )

        return final_score, {
            "extracted_answer": extracted_answer,
            "ground_truth": ground_truth,
            "format_score": format_score,
            "accuracy_score": accuracy_score,
            "final_score": final_score,
        }
