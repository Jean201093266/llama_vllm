from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.utils.checkpoint import (
    get_last_checkpoint,
    read_checkpoint_manifest,
    refresh_checkpoint_manifests,
    write_checkpoint_manifest,
)


def test_write_and_read_latest_checkpoint_manifest(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-12"
    ckpt.mkdir()
    write_checkpoint_manifest(str(tmp_path), "latest", str(ckpt))
    manifest = read_checkpoint_manifest(str(tmp_path), kind="latest")
    assert manifest is not None
    assert manifest["path"] == str(ckpt)
    assert manifest["step"] == 12


def test_get_last_checkpoint_prefers_manifest(tmp_path: Path) -> None:
    old_ckpt = tmp_path / "checkpoint-5"
    new_ckpt = tmp_path / "checkpoint-20"
    old_ckpt.mkdir()
    new_ckpt.mkdir()
    write_checkpoint_manifest(str(tmp_path), "latest", str(old_ckpt))
    assert get_last_checkpoint(str(tmp_path)) == str(old_ckpt)


def test_refresh_checkpoint_manifests_writes_latest_and_best(tmp_path: Path) -> None:
    latest = tmp_path / "checkpoint-30"
    best = tmp_path / "checkpoint-10"
    latest.mkdir()
    best.mkdir()
    refresh_checkpoint_manifests(str(tmp_path), latest_checkpoint=str(latest), best_checkpoint=str(best))
    latest_manifest = read_checkpoint_manifest(str(tmp_path), kind="latest")
    best_manifest = read_checkpoint_manifest(str(tmp_path), kind="best")
    assert latest_manifest is not None and latest_manifest["path"] == str(latest)
    assert best_manifest is not None and best_manifest["path"] == str(best)

