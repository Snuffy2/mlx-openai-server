"""
MLX model handlers for text, multimodal, image generation, and embeddings models.
"""

from app.handler.mlx_embeddings import MLXEmbeddingsHandler
from app.handler.mlx_lm import MLXLMHandler
from app.handler.mlx_vlm import MLXVLMHandler

# Optional mflux import - only available if flux extra is installed
try:
    from app.handler.mflux import MLXFluxHandler

    MFLUX_AVAILABLE = True
except ImportError:
    MLXFluxHandler = None
    MFLUX_AVAILABLE = False

__all__ = [
    "MLXLMHandler",
    "MLXVLMHandler",
    "MLXFluxHandler",
    "MLXEmbeddingsHandler",
    "MFLUX_AVAILABLE",
]
