"""Hilfsfunktionen des Emulator-Laufs (ohne DFC testbar)."""

import numpy as np

from thesis.emulator.run import _read_targets, _split


def test_split_findet_logits_und_features():
    outs = [np.zeros((4, 7, 7, 960)), np.zeros((4, 1, 1, 1000))]
    assert _split(outs) == (1, 0)
    assert _split([np.zeros((4, 1000))]) == (0, None)


def test_read_targets_nimmt_erste_wiederholung(tmp_path):
    (tmp_path / "predictions.csv").write_text(
        "image_id,label,top1,top5,latency_ms,repeat,out_sha1\n"
        "a.JPEG,1,5,5 1,1.0,0,x\n"
        "a.JPEG,1,7,7 1,1.0,1,y\n", encoding="utf-8")
    assert _read_targets(tmp_path) == {"a.JPEG": 5}
    (tmp_path / "predictions.csv").write_text("image_id,label,top1,top5\nb.JPEG,2,3,3 2\n", encoding="utf-8")
    assert _read_targets(tmp_path) == {"b.JPEG": 3}
