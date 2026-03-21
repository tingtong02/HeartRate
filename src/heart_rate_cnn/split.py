from __future__ import annotations

from dataclasses import dataclass

from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class SubjectSplit:
    train_subjects: list[str]
    test_subjects: list[str]


def train_test_subject_split(
    subject_ids: list[str],
    test_size: float = 0.3,
    random_seed: int = 42,
) -> SubjectSplit:
    unique_subjects = sorted(set(subject_ids))
    if len(unique_subjects) < 2:
        return SubjectSplit(train_subjects=unique_subjects, test_subjects=unique_subjects)

    train_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )
    return SubjectSplit(
        train_subjects=sorted(train_subjects),
        test_subjects=sorted(test_subjects),
    )
