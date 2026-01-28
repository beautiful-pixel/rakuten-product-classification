from pathlib import Path
from PIL import Image
import pandas as pd


def load_images_from_df(
    df: pd.DataFrame,
    image_dir: Path,
):
    """
    Load images from disk based on product identifiers stored in a DataFrame.

    Image files are expected to follow the naming convention:
    ``image_{imageid}_product_{productid}.jpg``.

    This function is intended for offline usage (training, evaluation,
    or analysis). Images are loaded into memory as ``PIL.Image.Image``
    objects and converted to RGB.

    Args:
        df (pd.DataFrame):
            DataFrame containing at least the following columns:
            - ``imageid``
            - ``productid``

        image_dir (Path):
            Root directory containing the image files.

    Returns:
        list[PIL.Image.Image]:
            List of loaded images in RGB format, ordered according to
            the rows of the input DataFrame.

    Raises:
        KeyError:
            If required columns (``imageid`` or ``productid``) are missing
            from the DataFrame.

        FileNotFoundError:
            If one or more image files cannot be found on disk.
    """
    if not {"imageid", "productid"}.issubset(df.columns):
        raise KeyError(
            "DataFrame must contain 'imageid' and 'productid' columns."
        )

    image_dir = Path(image_dir)

    images = []
    for _, row in df.iterrows():
        img_path = image_dir / f"image_{row.imageid}_product_{row.productid}.jpg"
        images.append(Image.open(img_path).convert("RGB"))

    return images
