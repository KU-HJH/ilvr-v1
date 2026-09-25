import json
import pytest
import torch
from evaluation import latent_image_trajectory as lt
from evaluation.latent_image_layout import LayoutTokenIds

T = LayoutTokenIds(80, 81, 83, 84, 91, 90, 92, (70, 71))


class FakePrepared:
    def __init__(self, identity):
        ids = torch.tensor([[1, 2, 81, 80, 80, 83, 5, 6, 84, 7, 8]])
        self.tensors = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                        "pixel_values": torch.zeros(2, 12), "image_grid_thw": torch.tensor([[1, 1, 2]])}
        self.prompt_token_length = ids.shape[1]
        self.prompt_identity = identity


@pytest.fixture
def patched(monkeypatch, tmp_path):
    (tmp_path / "images_comt" / "deletion").mkdir(parents=True)
    monkeypatch.setattr(lt, "_open_images", lambda row, root: (["img"], ("images_comt/deletion/1.png",)))
    monkeypatch.setattr(lt, "prepare_evaluation_input", lambda p, s, i: FakePrepared("abc"))
    return tmp_path


def eval_row(identity="abc", length=11):
    return {"index": 7, "prompt_identity": identity, "prompt_token_length": length,
            "generated_suffix_token_ids": [10, 91, 90, 90, 92, 70, 71, 5, 84], "match": True}


def test_builds_prompt_plus_suffix(patched):
    tr = lt.build_trajectory(None, {"image_input": ["x"]}, eval_row(), 7, patched, T,
                             latent_size=2)
    assert tr.input_ids.shape == (1, 20) and tr.prompt_length == 11
    assert tr.layout.blocks[0].start == 12 and tr.category == "deletion" and tr.eval_match


def test_identity_mismatch_is_fatal(patched):
    with pytest.raises(ValueError, match="prompt identity mismatch"):
        lt.build_trajectory(None, {"image_input": ["x"]}, eval_row(identity="zzz"), 7, patched, T,
                            latent_size=2)


def test_length_mismatch_is_fatal(patched):
    with pytest.raises(ValueError, match="prompt identity mismatch"):
        lt.build_trajectory(None, {"image_input": ["x"]}, eval_row(length=10), 7, patched, T,
                            latent_size=2)


def test_selection_and_eval_rows(tmp_path):
    sel = tmp_path / "m.json"
    sel.write_text(json.dumps({"selection": {"requested_indices": [3, 1], "requested_count": 2}}))
    assert lt.load_selection(sel) == (3, 1)
    rows = tmp_path / "e.jsonl"
    rows.write_text("\n".join(json.dumps({"index": i}) for i in (1, 3)))
    assert set(lt.load_eval_rows(rows)) == {1, 3}
