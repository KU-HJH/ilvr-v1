"""Literal offline answer rules from ILVR 13a67ed eval.py (see original-provenance.json)."""
import re

_yes_set = {"yes", "true", "a"}
_no_set = {"no", "false", "b"}

def _clean_answer_span(cand: str) -> str:
    """
    Keep only the first line, drop special tokens, then peel surrounding
    markdown/quotes/brackets and trailing punctuation (e.g. "**2**", "2.").
    """
    cand = re.split(r"[\n\r]", (cand or "").strip())[0]
    cand = _strip_special_tokens(cand)
    cand = re.sub(r"^[\s\*_`'\"“‘(\[{【]+", "", cand)
    cand = re.sub(r"[\s\*_`'\"”’)\]}】.。,，!！?？;；:：]+$", "", cand)
    return cand.strip()


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from the model output and return a simplified string.

    Priority rules:
      1) \\boxed{...} (take the last one)
      2) Match 'final answer (is): XXX' (case-insensitive, colon optional)
      3) Match 'answer is: XXX' or 'answer: XXX'
      4) If the output contains exactly one of yes/no/true/false/a/b
         (case-insensitive), return it directly
      5) Otherwise, fall back to the cleaned full text (useful for debugging)

    For patterns 2/3 the LAST occurrence is used, since the conclusion
    normally appears at the end of the response.
    """
    s = _strip_special_tokens(text or "").strip()

    boxed = re.findall(r"\\boxed\{([^{}]*)\}", s)
    if boxed:
        return _clean_answer_span(boxed[-1])

    patterns = [
        r"final\s*answer\s*(?:is)?\s*[:：]?\s*(.+)",
        r"answer\s*(?:is)?\s*[:：]\s*(.+)",
    ]

    for pat in patterns:
        matches = list(re.finditer(pat, s, flags=re.IGNORECASE))
        if matches:
            cand = _clean_answer_span(matches[-1].group(1))
            if cand:
                return cand

    tokens = re.findall(r"[A-Za-z]+", s.lower())
    only = [t for t in tokens if t in (_yes_set | _no_set)]
    if len(only) == 1:
        return only[0].capitalize() if only[0] in {"yes", "no"} else only[0]

    return _clean_answer_span(s)


def _strip_special_tokens(s: str) -> str:
    """
    Remove common special or end-of-sequence markers from decoded text.
    """
    end_markers = [
        "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>",
        "</s>", "<s>", "[/INST]", "[INST]",
        "<|latent_start|>", "<|latent_end|>", "<|latent_pad|>",
    ]
    for m in end_markers:
        s = s.replace(m, "")
    return s.strip()


def extract_assistant_content(text: str) -> str:
    """
    Extract only the assistant's content from the raw model output.

    Designed to be compatible with ChatML / Qwen / LLaMA-style templates.

    Priority:
      1) <|im_start|>assistant ... <|im_end|>
      2) <|assistant|> ... (until an end marker or end of text)
      3) Fallback: content after the last occurrence of 'assistant'
    """
    s = text or ""

    # 1) ChatML-style pattern
    m = re.search(
        r"<\|im_start\|>\s*assistant\s*(.*?)(?:<\|im_end\|>|$)",
        s,
        flags=re.S | re.I,
    )
    if m:
        return _strip_special_tokens(m.group(1))

    # 2) Qwen / other styles
    m = re.search(
        r"<\|assistant\|>\s*(.*?)(?:<\|im_end\|>|<\|endoftext\|>|"
        r"<\|eot_id\|>|<\|end\|>|$)",
        s,
        flags=re.S | re.I,
    )
    if m:
        return _strip_special_tokens(m.group(1))

    # 3) Fallback: last assistant segment
    m = re.search(
        r"(?:^|\n|\r)assistant\s*[:：]?\s*(.*)$",
        s,
        flags=re.S | re.I,
    )
    if m:
        return _strip_special_tokens(m.group(1))

    # If nothing matches, return cleaned full text
    return _strip_special_tokens(s)


def normalize_for_match(ans: str) -> str:
    """
    Normalize answers for matching: yes/no style tasks, numeric answers
    ("2" vs "2.0"), and case-insensitive text comparison.
    """
    a = _clean_answer_span(ans)
    low = a.lower()
    if low in _yes_set:
        return "Yes"
    if low in _no_set:
        return "No"
    try:
        f = float(a.replace(",", ""))
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, OverflowError):
        pass
    return low


def score_original(assistant_text, sample):
    """Original main's default branch, including missing/empty gold matching."""
    extracted = extract_final_answer(assistant_text)
    gold = sample.get("original_final_answer", "")
    prediction_normalized = normalize_for_match(extracted)
    gold_normalized = normalize_for_match(gold)
    return dict(extracted_final_answer=extracted, gold_final_answer=gold,
                prediction_normalized=prediction_normalized, gold_normalized=gold_normalized,
                match=prediction_normalized == gold_normalized)
