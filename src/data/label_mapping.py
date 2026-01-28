import os
import json
import hashlib
from pathlib import Path
from typing import Union, Optional
import numpy as np


def classes_fp(classes: Union[np.ndarray, list]) -> str:
    """
    Compute classes fingerprint (SHA256[:16]) with stable serialization.

    Args:
        classes: Array or list of class labels

    Returns:
        Fingerprint string (16 chars)
    """
    if isinstance(classes, np.ndarray):
        classes = classes.tolist()

    # Stable serialization (must match extract_classes.py - default separators)
    classes_json = json.dumps(classes)
    return hashlib.sha256(classes_json.encode("utf-8")).hexdigest()[:16]


def _get_canonical_json_path() -> Path:
    """Get canonical classes JSON path (with env var override support)."""
    # Allow override via environment variable
    env_path = os.environ.get("CANONICAL_CLASSES_JSON")
    if env_path:
        return Path(env_path)

    # Default path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "artifacts" / "canonical_classes.json"


def load_canonical_classes(path: Optional[Path] = None):
    """
    Load canonical classes from JSON file.

    Args:
        path: Optional path to canonical_classes.json (defaults to artifacts/)

    Returns:
        Tuple of (classes_array, classes_fp, num_classes)

    Raises:
        FileNotFoundError: If canonical classes file not found (single-line error)
    """
    if path is None:
        path = _get_canonical_json_path()

    if not path.exists():
        raise FileNotFoundError(f"Canonical classes not found at {path}. Run: python extract_classes.py")

    with open(path, "r") as f:
        data = json.load(f)

    classes_array = np.array(data["classes"], dtype=np.int64)
    fp = data["classes_fp"]
    num_classes = len(classes_array)

    # Sanity check
    if num_classes != 27:
        raise AssertionError(f"Expected 27 canonical classes but got {num_classes} from {path}")

    return classes_array, fp, num_classes


try:
    CANONICAL_CLASSES, CANONICAL_CLASSES_FP, NUM_CANONICAL_CLASSES = load_canonical_classes()
except FileNotFoundError as e:
    # Re-raise with single-line message for import-time safety
    err_msg = (
        "Canonical classes artifacts missing. Run: python extract_classes.py "
        "(or set CANONICAL_CLASSES_JSON=/path/to/canonical_classes.json)"
    )
    raise FileNotFoundError(err_msg) from e


def verify_canonical_classes(classes: np.ndarray, raise_on_error: bool = True) -> bool:
    """
    Verify that given classes match CANONICAL_CLASSES.

    Args:
        classes: Classes array to verify
        raise_on_error: If True, raise exception on mismatch

    Returns:
        True if matches, False otherwise

    Raises:
        AssertionError: If classes don't match and raise_on_error=True (single-line error)
    """
    matches = np.array_equal(classes, CANONICAL_CLASSES)

    if not matches and raise_on_error:
        fp_given = classes_fp(classes)
        msg = (
            f"Classes mismatch: expected_fp={CANONICAL_CLASSES_FP} got_fp={fp_given} "
            f"expected_len={NUM_CANONICAL_CLASSES} got_len={len(classes)}"
        )
        raise AssertionError(msg)

    return matches


def reorder_probs_to_canonical(
    probs: np.ndarray,
    model_classes: np.ndarray,
    canonical_classes: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reorder model probabilities to match canonical class order.

    If model was trained with different class order (e.g., LabelEncoder.fit(DEV) with
    different pandas sorting), this function reorders probs so that probs[:, i] corresponds
    to canonical_classes[i].

    Args:
        probs: Model predictions of shape (N, num_classes)
        model_classes: Classes array used during training (e.g., encoder.classes_)
        canonical_classes: Target class order (defaults to CANONICAL_CLASSES)

    Returns:
        Reordered probs with same shape

    Raises:
        AssertionError: If model_classes and canonical_classes have different elements (single-line)
    """
    if canonical_classes is None:
        canonical_classes = CANONICAL_CLASSES

    # Ensure int64 dtype for stable comparison
    model_classes = np.asarray(model_classes, dtype=np.int64)
    canonical_classes = np.asarray(canonical_classes, dtype=np.int64)

    # Validate same set of classes (dtype-safe set comparison)
    if not np.array_equal(np.sort(model_classes), np.sort(canonical_classes)):
        msg = (
            f"Model classes set mismatch: model_classes_len={len(model_classes)} "
            f"canonical_len={len(canonical_classes)}"
        )
        raise AssertionError(msg)

    # If already aligned, return as-is
    if np.array_equal(model_classes, canonical_classes):
        return probs

    # Build reorder mapping: canonical_idx -> model_idx
    model_to_idx = {int(cls): i for i, cls in enumerate(model_classes)}
    reorder_indices = [model_to_idx[int(cls)] for cls in canonical_classes]

    # Reorder columns
    probs_aligned = probs[:, reorder_indices]

    return probs_aligned


def encode_labels(
    y: Union[np.ndarray, list],
    canonical_classes: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Encode labels using canonical class order.

    Args:
        y: Raw class labels (e.g., [10, 2280, 50, ...])
        canonical_classes: Target class order (defaults to CANONICAL_CLASSES)

    Returns:
        Encoded labels (integers 0 to num_classes-1)

    Example:
        >>> y = [10, 2280, 10, 50]
        >>> encode_labels(y)
        array([0, 18, 0, 2])
    """
    if canonical_classes is None:
        canonical_classes = CANONICAL_CLASSES

    y = np.asarray(y, dtype=np.int64)

    # Build mapping: class -> index
    class_to_idx = {int(cls): i for i, cls in enumerate(canonical_classes)}

    # Encode
    encoded = np.array([class_to_idx[int(label)] for label in y], dtype=np.int64)

    return encoded


def decode_labels(
    y_encoded: Union[np.ndarray, list],
    canonical_classes: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Decode labels from canonical indices back to original class labels.

    Args:
        y_encoded: Encoded labels (integers 0 to num_classes-1)
        canonical_classes: Source class order (defaults to CANONICAL_CLASSES)

    Returns:
        Original class labels

    Example:
        >>> y_encoded = [0, 18, 0, 2]
        >>> decode_labels(y_encoded)
        array([10, 2280, 10, 50])
    """
    if canonical_classes is None:
        canonical_classes = CANONICAL_CLASSES

    y_encoded = np.asarray(y_encoded, dtype=np.int64)

    # Decode
    decoded = canonical_classes[y_encoded]

    return decoded


def print_canonical_info():
    """Print canonical classes information (compact)."""
    print("="*80)
    print("CANONICAL CLASSES (Project-Level Truth)")
    print("="*80)
    print(f"Classes_fp: {CANONICAL_CLASSES_FP} num_classes: {NUM_CANONICAL_CLASSES}")
    print("="*80)
