"""MLX Inference Service."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlx-inference-service")

except PackageNotFoundError:
    __version__ = "unknown"
