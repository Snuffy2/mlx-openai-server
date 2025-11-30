"""Image processing utilities for MLX OpenAI server."""

from __future__ import annotations

import asyncio
import contextlib
import gc
from io import BytesIO
from typing import Any

from loguru import logger
from PIL import Image

from .base_processor import BaseProcessor


class ImageProcessor(BaseProcessor):
    """Image processor for handling image files with caching, validation, and processing."""

    def __init__(self, max_workers: int = 4, cache_size: int = 1000) -> None:
        super().__init__(max_workers, cache_size)
        Image.MAX_IMAGE_PIXELS = 100000000  # Limit to 100 megapixels

    def _get_media_format(self, _media_url: str, _data: bytes | None = None) -> str:
        """Determine image format from URL or data."""
        # For images, we always save as PNG for consistency
        return "png"

    def _validate_media_data(self, data: bytes) -> bool:
        """Validate basic image data."""
        if len(data) < 100:  # Too small to be a valid image file
            return False

        # Check for common image file signatures
        image_signatures = [
            b"\xff\xd8\xff",  # JPEG
            b"\x89PNG\r\n\x1a\n",  # PNG
            b"GIF87a",  # GIF87a
            b"GIF89a",  # GIF89a
            b"BM",  # BMP
            b"II*\x00",  # TIFF (little endian)
            b"MM\x00*",  # TIFF (big endian)
            b"RIFF",  # WebP (part of RIFF)
        ]

        for sig in image_signatures:
            if data.startswith(sig):
                return True

        # Additional check for WebP
        if data.startswith(b"RIFF") and b"WEBP" in data[:20]:
            return True

        return False

    def _get_timeout(self) -> int:
        """Get timeout for HTTP requests."""
        return 30  # Standard timeout for images

    def _get_max_file_size(self) -> int:
        """Get maximum file size in bytes."""
        return 100 * 1024 * 1024  # 100 MB limit for images

    def _get_media_type_name(self) -> str:
        """Get media type name for logging."""
        return "image"

    def _resize_image_keep_aspect_ratio(
        self,
        image: Image.Image,
        max_size: int = 448,
    ) -> Image.Image:
        """
        Resize an image so its larger dimension does not exceed max_size while preserving aspect ratio.
        
        If both width and height are already less than or equal to max_size, the original image is returned unchanged.
        
        Parameters:
            image (PIL.Image.Image): Source image to resize.
            max_size (int): Maximum allowed size (pixels) for the image's larger dimension.
        
        Returns:
            PIL.Image.Image: The resized image, or the original image if no resizing was needed.
        """
        width, height = image.size
        if width <= max_size and height <= max_size:
            return image
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"Resized image to {new_width}x{new_height} from {width}x{height}")

        return image

    def _prepare_image_for_saving(self, image: Image.Image) -> Image.Image:
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "RGBA":
                background.paste(image, mask=image.split()[3])
            else:
                background.paste(image, mask=image.split()[1])
            return background
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def _process_media_data(self, data: bytes, cached_path: str, **kwargs: Any) -> str:
        """
        Process raw image bytes, optionally resize, prepare for saving as an RGB PNG, write the file to the given cache path, and trigger cache cleanup.
        
        Parameters:
            data (bytes): Raw image file bytes to process.
            cached_path (str): Filesystem path where the processed PNG will be written.
            resize (bool, optional): If True (default), resize the image to fit the configured maximum while preserving aspect ratio. Provided via kwargs.
        
        Returns:
            str: The path to the saved cached image.
        """
        image = None
        resize = kwargs.get("resize", True)
        try:
            with Image.open(BytesIO(data), mode="r") as image:
                if resize:
                    image = self._resize_image_keep_aspect_ratio(image)
                image = self._prepare_image_for_saving(image)
                image.save(cached_path, "PNG", quality=100, optimize=True)

            self._cleanup_old_files()
            return cached_path
        finally:
            # Ensure image object is closed to free memory
            if image:
                with contextlib.suppress(Exception):
                    image.close()

    async def process_image_url(self, image_url: str, *, resize: bool = True) -> str:
        """
        Process an image URL, cache the resulting PNG file, and return the cached file path.
        
        Parameters:
            resize (bool): If True, the image will be resized to fit within the processor's maximum size before saving.
        
        Returns:
            cached_path (str): Filesystem path to the saved cached image.
        """
        return await self._process_single_media(image_url, resize=resize)

    async def process_image_urls(
        self,
        image_urls: list[str],
        *,
        resize: bool = True,
    ) -> list[str | BaseException]:
        """
        Process a batch of image URLs and cache each image locally.
        
        Parameters:
            image_urls (list[str]): Iterable of image URLs to process.
            resize (bool): When True, resize each image according to the processor's resizing policy.
        
        Returns:
            list[str | BaseException]: A list where each element is the path to the cached file for a successfully processed URL, or a `BaseException` instance for a URL that failed.
        """
        tasks = [self.process_image_url(url, resize=resize) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Force garbage collection after batch processing
        gc.collect()
        return results