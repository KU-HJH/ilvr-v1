import pytest

from scripts.select_jsonl_rows import select_jsonl_rows


def test_selector_preserves_requested_order_and_uses_exactly_one_lf(tmp_path):
    """Catches sorting rows or leaking CRLF/missing-final-LF into the subset."""
    source = tmp_path / "source.jsonl"
    output = tmp_path / "nested" / "subset.jsonl"
    source.write_bytes(b'{"row":0}\r\n{"row":1}\n{"row":2}')

    select_jsonl_rows(source, output, (2, 0))

    assert output.read_bytes() == b'{"row":2}\n{"row":0}\n'


@pytest.mark.parametrize("indices", [(), (0, 0), (-1,), (3,)])
def test_selector_rejects_invalid_indices_without_publishing(tmp_path, indices):
    """Catches ambiguous, negative, or out-of-range subset publications."""
    source = tmp_path / "source.jsonl"
    output = tmp_path / "subset.jsonl"
    source.write_bytes(b"zero\none\ntwo\n")

    with pytest.raises(ValueError):
        select_jsonl_rows(source, output, indices)

    assert not output.exists()


def test_selector_refuses_to_overwrite_existing_output(tmp_path):
    """Catches a rerun replacing an earlier exact source witness."""
    source = tmp_path / "source.jsonl"
    output = tmp_path / "subset.jsonl"
    source.write_bytes(b"new\n")
    output.write_bytes(b"preserve\n")

    with pytest.raises(ValueError, match="already exists"):
        select_jsonl_rows(source, output, (0,))

    assert output.read_bytes() == b"preserve\n"
