import pytest

from environ.file_system import sample_csv


def test_sample_csv_returns_exception_on_no_file():
    with pytest.raises(FileNotFoundError):
        sample_csv("/tmp/no_file.csv", 1, 1)
