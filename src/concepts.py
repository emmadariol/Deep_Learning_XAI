"""AwA2 semantic-attribute utilities for concept-level explainability."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AwA2ConceptBank:
    """Class-level AwA2 semantic attributes."""

    class_names: list[str]
    concept_names: list[str]
    matrix: np.ndarray

    def normalized_matrix(self) -> np.ndarray:
        """Return concept values scaled to [0, 1] per concept."""
        values = self.matrix.astype(np.float64)
        mins = values.min(axis=0, keepdims=True)
        maxs = values.max(axis=0, keepdims=True)
        return (values - mins) / np.maximum(maxs - mins, 1e-12)

    def concept_vector(self, class_name: str, normalized: bool = True) -> np.ndarray:
        index = self.class_to_index()[normalize_class_name(class_name)]
        matrix = self.normalized_matrix() if normalized else self.matrix
        return matrix[index]

    def class_to_index(self) -> dict[str, int]:
        return {normalize_class_name(name): index for index, name in enumerate(self.class_names)}


def normalize_class_name(name: str) -> str:
    """Normalize class names across manifests and AwA2 metadata files."""
    return name.strip().lower().replace("+", "_").replace(" ", "_")


def read_manifest_classes(manifest_path: str | Path) -> list[str]:
    """Return manifest class names ordered by integer label."""
    by_label: dict[int, str] = {}
    with Path(manifest_path).expanduser().resolve().open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            by_label[int(row["label"])] = row["class_name"]
    if not by_label:
        raise ValueError(f"No classes found in manifest: {manifest_path}")
    return [by_label[index] for index in sorted(by_label)]


def find_awa2_metadata_root(paths: list[str | Path]) -> Path:
    """Find the folder containing AwA2 classes/predicates/attribute matrix files."""
    required_any = (
        "predicate-matrix-continuous.txt",
        "predicate-matrix-binary.txt",
    )
    required_all = ("classes.txt", "predicates.txt")

    def is_metadata_root(candidate: Path) -> bool:
        return all((candidate / name).exists() for name in required_all) and any(
            (candidate / name).exists() for name in required_any
        )

    for raw_path in paths:
        root = Path(raw_path).expanduser().resolve()
        candidates = [
            root,
            root / "Animals_with_Attributes2",
            root / "AwA2" / "Animals_with_Attributes2",
        ]
        for candidate in candidates:
            if candidate.is_dir() and is_metadata_root(candidate):
                return candidate

        for candidate in root.iterdir() if root.is_dir() else []:
            if candidate.is_dir() and is_metadata_root(candidate):
                return candidate

    raise FileNotFoundError(
        "Could not find AwA2 metadata files. Expected classes.txt, predicates.txt, "
        "and predicate-matrix-continuous.txt or predicate-matrix-binary.txt."
    )


def read_indexed_names(path: Path) -> list[str]:
    """Read AwA2 files with rows like '<index> <name>'."""
    names: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) == 1:
                names.append(parts[0])
            else:
                names.append(parts[1])
    return names


def load_awa2_concepts(
    metadata_root: str | Path,
    matrix_kind: str = "continuous",
) -> AwA2ConceptBank:
    """Load AwA2 class-level semantic attributes."""
    root = Path(metadata_root).expanduser().resolve()
    classes_path = root / "classes.txt"
    predicates_path = root / "predicates.txt"
    if matrix_kind == "continuous":
        matrix_path = root / "predicate-matrix-continuous.txt"
    elif matrix_kind == "binary":
        matrix_path = root / "predicate-matrix-binary.txt"
    else:
        raise ValueError("matrix_kind must be 'continuous' or 'binary'.")

    if not matrix_path.exists() and matrix_kind == "continuous":
        matrix_path = root / "predicate-matrix-binary.txt"
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)

    class_names = read_indexed_names(classes_path)
    concept_names = read_indexed_names(predicates_path)
    matrix = np.loadtxt(matrix_path, dtype=np.float64)

    if matrix.shape != (len(class_names), len(concept_names)):
        raise ValueError(
            f"Attribute matrix shape {matrix.shape} does not match "
            f"{len(class_names)} classes x {len(concept_names)} concepts."
        )

    return AwA2ConceptBank(
        class_names=class_names,
        concept_names=concept_names,
        matrix=matrix,
    )


def align_concept_bank_to_manifest(
    concept_bank: AwA2ConceptBank,
    manifest_class_names: list[str],
) -> AwA2ConceptBank:
    """Reorder concept bank rows to match manifest label order."""
    class_to_index = concept_bank.class_to_index()
    indices: list[int] = []
    missing: list[str] = []
    for class_name in manifest_class_names:
        key = normalize_class_name(class_name)
        if key not in class_to_index:
            missing.append(class_name)
        else:
            indices.append(class_to_index[key])

    if missing:
        raise ValueError(f"Manifest classes missing from AwA2 metadata: {missing}")

    return AwA2ConceptBank(
        class_names=manifest_class_names,
        concept_names=concept_bank.concept_names,
        matrix=concept_bank.matrix[np.array(indices)],
    )


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first, second) / denominator)


def top_concepts_for_class(
    concept_bank: AwA2ConceptBank,
    class_name: str,
    top_k: int = 12,
    normalized: bool = True,
) -> list[tuple[str, float]]:
    """Return strongest concepts for one class."""
    vector = concept_bank.concept_vector(class_name, normalized=normalized)
    indices = np.argsort(vector)[::-1][:top_k]
    return [(concept_bank.concept_names[index], float(vector[index])) for index in indices]


def concept_transition_summary(
    concept_bank: AwA2ConceptBank,
    source_class: str,
    target_class: str,
    top_k: int = 8,
) -> dict[str, object]:
    """Summarize concept changes between two class-level attribute vectors."""
    source = concept_bank.concept_vector(source_class, normalized=True)
    target = concept_bank.concept_vector(target_class, normalized=True)
    delta = target - source
    gained_indices = [
        int(index) for index in np.argsort(delta)[::-1] if delta[index] > 0
    ][:top_k]
    lost_indices = [
        int(index) for index in np.argsort(delta) if delta[index] < 0
    ][:top_k]
    shared_indices = np.argsort(np.minimum(source, target))[::-1][:top_k]

    return {
        "source_class": source_class,
        "target_class": target_class,
        "concept_cosine": cosine_similarity(source, target),
        "mean_abs_concept_delta": float(np.mean(np.abs(delta))),
        "gained_concepts": "; ".join(
            f"{concept_bank.concept_names[index]}:{delta[index]:.3f}" for index in gained_indices
        ),
        "lost_concepts": "; ".join(
            f"{concept_bank.concept_names[index]}:{-delta[index]:.3f}" for index in lost_indices
        ),
        "shared_concepts": "; ".join(
            f"{concept_bank.concept_names[index]}:{min(source[index], target[index]):.3f}"
            for index in shared_indices
        ),
    }
