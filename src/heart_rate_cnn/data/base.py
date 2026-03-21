from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from heart_rate_cnn.types import SubjectRecord


class BaseLoader(ABC):
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def validate_root(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.root_dir}")

    @abstractmethod
    def list_subjects(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def load_subject(self, subject_id: str) -> SubjectRecord:
        raise NotImplementedError
