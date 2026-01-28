from pathlib import Path
import pandas as pd


def get_image_paths(
    df: pd.DataFrame,
    image_dir: Path,
) -> pd.Series:
    """
    Build image file paths from product identifiers.

    Image filenames are expected to follow the convention:
    ``image_{imageid}_product_{productid}.jpg``.

    This utility function is intended for offline dataset usage
    (e.g. training, evaluation, or analysis) where images are stored
    on disk and referenced by their identifiers.

    Args:
        df (pd.DataFrame):
            DataFrame containing at least the following columns:
            - ``imageid``
            - ``productid``

        image_dir (Path):
            Root directory containing the image files.

    Returns:
        pd.Series:
            Series of ``Path`` objects pointing to image files on disk.

    Raises:
        KeyError:
            If required columns (``imageid`` or ``productid``) are missing
            from the DataFrame.
    """
    if not {"imageid", "productid"}.issubset(df.columns):
        raise KeyError(
            "DataFrame must contain 'imageid' and 'productid' columns."
        )

    file_names = (
        "image_" + df["imageid"].astype(str) +
        "_product_" + df["productid"].astype(str) + ".jpg"
    )

    image_dir = Path(image_dir)

    return file_names.apply(lambda name: image_dir / name)
