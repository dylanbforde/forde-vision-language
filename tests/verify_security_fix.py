import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock dependencies that might be missing or slow
sys.modules["datasets"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["tqdm"] = MagicMock()
sys.modules["tqdm.auto"] = MagicMock()
sys.modules["src.data.dataset"] = MagicMock()

import src.data.download_to_drive as download_to_drive

def test_download_and_save_uses_tempfile():
    output_dir = "test_output"

    # Setup mocks
    mock_dataset = MagicMock()
    # Mock the iterator to yield 3 examples
    mock_dataset.__iter__.return_value = iter([{"caption": "test", "image_url": "url"}] * 3)
    mock_dataset.skip.return_value = mock_dataset
    mock_dataset.take.return_value = mock_dataset

    download_to_drive.datasets.load_dataset.return_value = mock_dataset

    with patch("src.data.download_to_drive.tempfile.TemporaryDirectory") as mock_tempdir:
        mock_tempdir.return_value.__enter__.return_value = "/tmp/fake_temp_dir"

        mock_shard = MagicMock()
        download_to_drive.datasets.Dataset.from_list.return_value = mock_shard

        with patch("shutil.copytree") as mock_copytree:
            with patch("os.path.exists", return_value=False):
                # Run with shard_size=2 to trigger one shard save and one final shard save
                # We need to mock tqdm as well
                with patch("src.data.download_to_drive.tqdm", side_effect=lambda x, **kwargs: x):
                    download_to_drive.download_and_save(output_dir, shard_size=2)

                    # Verify TemporaryDirectory was called at least twice
                    # 1 for shard 0, 1 for shard 1 (final)
                    assert mock_tempdir.call_count >= 2

                    # Verify save_to_disk was called with the temp dir
                    mock_shard.save_to_disk.assert_called_with("/tmp/fake_temp_dir")

                    # Verify copytree was called from temp dir to final dir
                    # Should be called for shard_0 and shard_1
                    expected_calls = [
                        (("/tmp/fake_temp_dir", os.path.join(output_dir, "shard_0")),),
                        (("/tmp/fake_temp_dir", os.path.join(output_dir, "shard_1")),),
                    ]
                    for call_args in expected_calls:
                        mock_copytree.assert_any_call(*call_args[0])

    print("Verification successful!")

if __name__ == "__main__":
    try:
        test_download_and_save_uses_tempfile()
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
