import re
from fractions import Fraction
from typing import Any, Dict, List, Optional, Union

NumberLike = Union[int, float, str]

def _strip_end_token(response: str, end_token: Optional[str]) -> str:
    if end_token and response.endswith(end_token):
        return response[: -len(end_token)]
    return response

def _clean_numeric_token(tok: str) -> str:
    tok = tok.strip()

    while tok and not (tok[0].isdigit() or tok[0] in "+-"):
        tok = tok[1:]

    while tok and not tok[-1].isdigit():
        if tok[-1] in ".,;:)]}":
            tok = tok[:-1]
        else:
            tok = tok[:-1]

    tok = tok.replace(",", "")
    return tok

def extract_final_numeric_answer(response: str) -> Optional[str]:
    matches = re.findall(r"####\s*([^\n]+)", response)
    if not matches:
        return None

    raw = matches[-1].strip()
    parts = raw.split()
    if not parts:
        return None

    first_token = parts[0].strip()
    cleaned = _clean_numeric_token(first_token)

    if cleaned == "":
        return None
    return cleaned

def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    response = _strip_end_token(response, end_token)

    full_format_regex = r"^<think>.*?</think>.*?####\s*\S+.*$"

    think_match = re.search(r"<think>.*?</think>", response, re.DOTALL)
    answer_marker_match = re.search(r"####\s*\S+", response)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0
    if think_match:
        reward += 0.1
    if answer_marker_match:
        reward += 0.5

    return reward

def answer_reward_function(
    response: str,
    numbers: List[int] = None,
    target: NumberLike = None,
) -> float:
    if target is None:
        return 0.0

    pred_ans = extract_final_numeric_answer(response)
    if pred_ans is None:
        return 0.0

    target_str = _clean_numeric_token(str(target))
    if target_str == "":
        return 0.0

    try:
        def to_float(s: str) -> float:
            if "/" in s and not any(ch in s for ch in ".eE"):
                return float(Fraction(s))
            return float(s)

        pred_val = to_float(pred_ans)
        tgt_val = to_float(target_str)

        if abs(pred_val - tgt_val) < 1e-6:
            return 1.0
        else:
            return 0.0
    except Exception:
        return 1.0 if pred_ans == target_str else 0.0

def reward_function(
    response: str,
    numbers: List[int] = None,
    target: NumberLike = None,
    end_token: str = None,
) -> Dict[str, Any]:
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, numbers, target)

    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }