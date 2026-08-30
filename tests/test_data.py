import json

from train.data import fmt, load_texts


def test_fmt_wraps_text():
    out = fmt("hello")
    assert out["text"].startswith("Проанализируй")
    assert "hello" in out["text"]
    assert out["text"].endswith("Главное:")


def test_load_texts_skips_bad_lines(tmp_path):
    f = tmp_path / "batch.jsonl"
    f.write_text(
        json.dumps({"text": "good"}) + "\n"
        + "not json\n"
        + json.dumps({"no_text": 1}) + "\n"
        + json.dumps({"text": "second"}) + "\n",
        encoding="utf-8",
    )
    assert load_texts(str(tmp_path)) == ["good", "second"]


def test_load_texts_empty_dir(tmp_path):
    assert load_texts(str(tmp_path)) == []