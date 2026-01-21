"""Logging utilities."""

import logging

from rich.logging import RichHandler

# Level color map
DEFAULT_COLOR = ("#D3D7CF", "#007166")

COLOR_MAP = {
    "INFO": DEFAULT_COLOR,
    "DEBUG": ("#D3D7CF", "#4084C4"),
    "WARNING": ("#D3D7CF", "#E67E22"),
    "ERROR": ("#D3D7CF", "#E74C3C"),
    "FATAL": ("#D3D7CF", "#C0392B"),
    "EXC": ("#D3D7CF", "#8E44AD"),
}


class StyleFormatter(logging.Formatter):
    """
    Custom logging formatter that adds color-coded level badges with Rich markup.
    """

    def __init__(self, use_level_colors: bool = False):
        super().__init__()
        self.use_level_colors = use_level_colors
        self.default_colors = DEFAULT_COLOR
        self.column_width = 9

    def _format_level_badge(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname

        is_exc = bool(record.exc_info) or bool(getattr(record, "exc_text", None))

        if is_exc and original_levelname in {"ERROR", "CRITICAL"}:
            display_levelname = "EXC"
        elif original_levelname == "CRITICAL":
            display_levelname = "FATAL"
        else:
            display_levelname = original_levelname

        if self.use_level_colors:
            text_color, bg_color = COLOR_MAP.get(display_levelname, self.default_colors)
        else:
            text_color, bg_color = self.default_colors

        label = f" {display_levelname} "
        style_def = f"{text_color} on {bg_color}" if bg_color else f"{text_color}"

        pad_len = max(0, self.column_width - len(label))
        padding = " " * pad_len

        return f"{padding}[{style_def}]{label}[/{style_def}]"

    def formatMessage(self, record: logging.LogRecord) -> str:
        badge = self._format_level_badge(record)
        return f"  {badge}  {record.message}"

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        return self.formatMessage(record)


def setup_logging(
    level: int | str = logging.INFO,
    name: str | None = "app",
    use_level_colors: bool = False,
):
    """
    Configure a logger with RichHandler and StyleFormatter.

    Args:
        level: Logging level constant or string.
        name: Logger name to configure. Use None for root logger.
        use_level_colors: If True, use different colors for each log level.
    """

    rich_handler = RichHandler(
        show_time=False,
        show_path=False,
        show_level=False,
        markup=True,
        rich_tracebacks=True,
    )

    rich_handler.setFormatter(StyleFormatter(use_level_colors=use_level_colors))

    if name is None:
        logging.basicConfig(level=level, handlers=[rich_handler], force=True)

    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers = [rich_handler]
        logger.propagate = False
