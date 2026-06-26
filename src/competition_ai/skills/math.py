from collections import Counter
from typing import List, Any

class MathSkill:
    """
    Skill: Mathematical Problem Solving logic.
    Extracted from AIMO/Nemo kernels.
    """
    @staticmethod
    def select_consensus_answer(answers: List[Any], threshold: int = 4) -> Any:
        """
        Early stopping / Majority voting logic.
        If any answer appears >= threshold times, pick it.
        """
        if not answers:
            return None
        counts = Counter(answers)
        most_common = counts.most_common(1)[0]
        if most_common[1] >= threshold:
            return most_common[0]
        return most_common[0] # Fallback to most frequent

    @staticmethod
    def format_imo_prompt(problem: str) -> str:
        """Elite mathematical solver persona prompting."""
        return (
            "You are an elite mathematical problem solver. "
            "Solve the following problem step-by-step: "
            f"{problem}"
        )
