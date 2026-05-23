"""Template-based synthetic augmentation when frontier API is unavailable.

Generates real structured training rows from failure examples using
deterministic paraphrase templates — not random mock scores.
"""

from __future__ import annotations

import copy
import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

_TEMPLATES = [
    "Solve the following problem step by step: {q}",
    "Find the value that satisfies: {q}",
    "Work through this carefully: {q}",
    "Determine the answer to: {q}",
]


def _vary_numbers(text: str, rng: random.Random) -> str:
    """Replace integers with small offsets to create variations."""
    def repl(m: re.Match) -> str:
        n = int(m.group(0))
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        return str(max(0, n + delta))

    return re.sub(r"\b\d+\b", repl, text, count=min(3, len(re.findall(r"\b\d+\b", text))))


def generate_template_synthetic(
    failure_examples: Dict[str, List[Dict[str, Any]]],
    per_mode: int = 200,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build synthetic dataset from failure examples without external API."""
    rng = random.Random(seed)
    ts = datetime.now(timezone.utc).isoformat()
    output: List[Dict[str, Any]] = []

    for mode, examples in failure_examples.items():
        if not examples:
            continue
        for i in range(per_mode):
            base = copy.deepcopy(rng.choice(examples))
            q = base.get("question", "Solve for x.")
            varied_q = _vary_numbers(q, rng)
            template = rng.choice(_TEMPLATES)
            question = template.format(q=varied_q)

            ans = base.get("answer", "\\boxed{0}")
            if "\\boxed" not in str(ans):
                ans = f"\\boxed{{{ans}}}"

            thinking = (
                f"<<thinking>>\n"
                f"Target failure mode: {mode}.\n"
                f"Apply the same reasoning pattern as the reference problem.\n"
                f"</thinking>>"
            )

            output.append({
                "question": question,
                "thinking_trace": thinking,
                "answer": ans,
                "failure_mode_tag": mode,
                "difficulty_estimate": min(1.0, 0.3 + (i % 7) * 0.1),
                "generation_timestamp": ts,
                "source_model": "template_augmentation",
            })

    return output
