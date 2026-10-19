"""Validate uploads without expanding their encoded representation."""
import io
import warnings

from PIL import Image, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile


def normalize_image(image: UploadedFile) -> bytes:
    if not image.data or len(image.data) > 20 * 1024 * 1024:
        raise CommandError("invalid_image", "Upload an image between 1 byte and 20 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image.data)) as source:
                if source.format not in ("PNG", "JPEG", "WEBP"):
                    raise ValueError("Unsupported format")
                if min(source.size) < 64 or source.width * source.height > 20_000_000:
                    raise ValueError("Invalid dimensions")
                source.load()
                return image.data
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise CommandError("invalid_image", "Use a valid PNG, JPEG or WebP, at least 64 pixels per side and at most 20 megapixels.") from exc
