from unittest.mock import MagicMock

from src.data import download_to_drive


class _FakeStream:
    def __iter__(self):
        return iter(
            [
                {"caption": "caption 1", "image_url": "url-1"},
                {"caption": "caption 2", "image_url": "url-2"},
                {"caption": "caption 3", "image_url": "url-3"},
            ]
        )

    def skip(self, _count):
        return self

    def take(self, _count):
        return self


class _FakeTokenizer:
    def __call__(self, *_args, **_kwargs):
        return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}


class _FakeTemporaryDirectory:
    calls = 0

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        type(self).calls += 1
        return f"/tmp/fake-shard-{type(self).calls}"

    def __exit__(self, *_args):
        return False


def test_download_and_save_uses_managed_temporary_directories(tmp_path, monkeypatch):
    _FakeTemporaryDirectory.calls = 0
    fake_shard = MagicMock()

    monkeypatch.setattr(
        download_to_drive.datasets,
        "load_dataset",
        lambda *_, **__: _FakeStream(),
    )
    monkeypatch.setattr(
        download_to_drive.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTokenizer(),
    )
    monkeypatch.setattr(download_to_drive, "process_image", lambda _url: "image")
    monkeypatch.setattr(
        download_to_drive.datasets.Dataset,
        "from_list",
        lambda *_, **__: fake_shard,
    )
    monkeypatch.setattr(
        download_to_drive.tempfile,
        "TemporaryDirectory",
        _FakeTemporaryDirectory,
    )
    monkeypatch.setattr(download_to_drive.shutil, "copytree", MagicMock())
    monkeypatch.setattr(download_to_drive, "tqdm", lambda iterable, **_kwargs: iterable)

    download_to_drive.download_and_save(str(tmp_path), num_proc=1, shard_size=2)

    assert _FakeTemporaryDirectory.calls == 2
    fake_shard.save_to_disk.assert_any_call("/tmp/fake-shard-1")
    fake_shard.save_to_disk.assert_any_call("/tmp/fake-shard-2")
