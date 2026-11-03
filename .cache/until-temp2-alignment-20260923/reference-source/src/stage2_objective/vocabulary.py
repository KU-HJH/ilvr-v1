"""Construction of the exact text-only scoring vocabulary."""

from dataclasses import dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class TextVocabulary:
    output_size: int
    text_ids: tuple[int, ...]
    excluded: tuple[tuple[int, str, str], ...]
    identity: str


_CONFIGURED_TOKENS = {
    "image_token_id": ("<|image_pad|>", "<image>"),
    "video_token_id": ("<|video_pad|>", "<video>"),
    "vision_start_token_id": ("<|vision_start|>",),
    "vision_end_token_id": ("<|vision_end|>",),
    "latent_token_id": ("<|latent_pad|>",),
    "latent_pad_token_id": ("<|latent_pad|>",),
    "latent_start_id": ("<|latent_start|>",),
    "latent_start_token_id": ("<|latent_start|>",),
    "latent_end_id": ("<|latent_end|>",),
    "latent_end_token_id": ("<|latent_end|>",),
}


def _token_for_id(tokenizer: Any, vocab: dict[str, int], token_id: int) -> str:
    reverse = {value: token for token, value in vocab.items()}
    token = tokenizer.convert_ids_to_tokens(token_id) if hasattr(tokenizer, "convert_ids_to_tokens") else reverse.get(token_id)
    if token != reverse.get(token_id):
        raise ValueError(f"token ID {token_id} does not round-trip through tokenizer vocabulary")
    return token


def build_text_vocabulary(
    tokenizer: Any,
    model_config: Any,
    *,
    embedding_rows: int,
    output_rows: int,
) -> TextVocabulary:
    if type(embedding_rows) is not int or type(output_rows) is not int or min(embedding_rows, output_rows) <= 0:
        raise ValueError("vocabulary row counts must be positive integers")
    configured_rows = getattr(model_config, "vocab_size", None)
    if type(configured_rows) is not int or configured_rows != embedding_rows or embedding_rows != output_rows:
        raise ValueError("configured, input, and output vocabulary rows must agree")
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or any(not isinstance(token, str) or type(token_id) is not int for token, token_id in vocab.items()):
        raise TypeError("tokenizer vocabulary must map strings to integer IDs")
    known_ids = set(vocab.values())
    if len(known_ids) != len(vocab):
        raise ValueError("tokenizer vocabulary IDs must be unique")
    if known_ids != set(range(output_rows)):
        raise ValueError("unexplained tokenizer/output vocabulary mismatch")

    records: set[tuple[int, str, str]] = set()
    for token_id in getattr(tokenizer, "all_special_ids", ()):
        if type(token_id) is not int or token_id not in known_ids:
            raise ValueError("tokenizer special ID is outside the output vocabulary")
        records.add((token_id, _token_for_id(tokenizer, vocab, token_id), "tokenizer special token"))
    for token_id, added in getattr(tokenizer, "added_tokens_decoder", {}).items():
        if getattr(added, "special", False):
            if type(token_id) is not int or token_id not in known_ids:
                raise ValueError("special AddedToken ID is outside the output vocabulary")
            content = getattr(added, "content", None)
            actual = _token_for_id(tokenizer, vocab, token_id)
            if content != actual:
                raise ValueError("special AddedToken does not round-trip to its intended token")
            records.add((token_id, actual, "special AddedToken"))

    for field, intended_tokens in _CONFIGURED_TOKENS.items():
        token_id = getattr(model_config, field, None)
        if token_id is None:
            continue
        if type(token_id) is not int or token_id not in known_ids:
            raise ValueError(f"configured {field} is outside the output vocabulary")
        actual = _token_for_id(tokenizer, vocab, token_id)
        if actual not in intended_tokens:
            raise ValueError(f"configured {field} does not map to its intended token")
        kind = "latent" if "latent" in field else field.removesuffix("_token_id").replace("_", " ")
        records.add((token_id, actual, f"configured {kind} token"))

    exclusion_records = sorted(records)
    excluded_ids = {token_id for token_id, _, _ in exclusion_records}
    text_ids = tuple(sorted(known_ids - excluded_ids))
    if not text_ids:
        raise ValueError("text vocabulary is empty")
    identity = hashlib.sha256(json.dumps(
        {"vocab": sorted(vocab.items()), "excluded": exclusion_records},
        separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    return TextVocabulary(output_rows, text_ids, tuple(exclusion_records), identity)
