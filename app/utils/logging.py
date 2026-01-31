"""Logging utilities."""

import logging
from collections.abc import Iterable

from rich.console import Console
from rich.text import Text
from rich.traceback import Traceback

# Badge background colors
DEFAULT_BADGE_COLOR = "#007166"

BADGE_COLOR_MAP = {
    "INFO": "#007166",
    "DEBUG": "#365f8c",
    "WARNING": "#c06d1a",
    "ERROR": "#a93226",
    "FATAL": "#c64535",
    "EXC": "#6f3b8a",
    "EXTRA": "#2b7a78",
}

# Compute standard LogRecord keys once for fast filtering
_STANDARD_RECORD_KEYS = set(
    logging.LogRecord(
        name="x",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__.keys()
)

# Default extra prefix
_EXTRA_PREFIX: str | None = "ctx_"


def ctx(**kwargs: object) -> dict[str, object]:
    """Build structured logging context with configured prefix."""

    prefix = _EXTRA_PREFIX or ""
    return {f"{prefix}{k}": v for k, v in kwargs.items()}


class StyledRichHandler(logging.Handler):
    """
    Rich console handler that prints the formatter output
    and prints Rich tracebacks separately when exc_info is present.
    """

    def __init__(self) -> None:
        super().__init__()
        self.console = Console(stderr=True, soft_wrap=True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Render with our formatter
            msg = self.format(record)

            # Soft wrap
            self.console.print(
                Text.from_markup(msg),
                soft_wrap=True,
            )

            # If there's a real exception, print a Rich traceback block
            exc_info = record.exc_info
            if exc_info and exc_info[0] is not None:
                tb = Traceback.from_exception(
                    exc_info[0],
                    exc_info[1],
                    exc_info[2],
                    show_locals=False,
                )
                self.console.print(tb)

        except Exception:
            self.handleError(record)


class StyleFormatter(logging.Formatter):
    """
    Formatter with Rich-markup level badge and optional structured extras.
    """

    def __init__(
        self,
        *,
        use_level_colors: bool = False,
        show_extra: bool = False,
        extra_prefix: str | None = None,
        extra_allowlist: set[str] | None = None,
        extra_multiline: bool = False,
    ):
        super().__init__()

        self.use_level_colors = use_level_colors
        self.column_width = 9

        self.show_extra = show_extra
        self.extra_allowlist = extra_allowlist
        self.extra_multiline = extra_multiline
        self.extra_prefix = _EXTRA_PREFIX if extra_prefix is None else extra_prefix
        self.max_extra_len = 200

    def _format_level_badge(self, record: logging.LogRecord) -> str:
        """Format the level badge with optional color."""

        original_levelname = record.levelname

        # Only treat as exception if we have a real exception type
        exc_info = record.exc_info
        is_exc = bool(exc_info and exc_info[0] is not None) or bool(
            getattr(record, "exc_text", None)
        )

        if is_exc and record.levelno >= logging.ERROR:
            display_levelname = "EXC"
        elif record.levelno >= logging.CRITICAL:
            display_levelname = "FATAL"
        else:
            display_levelname = original_levelname

        if self.use_level_colors:
            bg_color = BADGE_COLOR_MAP.get(display_levelname, DEFAULT_BADGE_COLOR)
        else:
            bg_color = DEFAULT_BADGE_COLOR

        label = f" {display_levelname} "
        style_def = f"white on {bg_color}"

        pad_len = max(0, self.column_width - len(label))
        padding = " " * pad_len

        return f"{padding}[{style_def}]{label}[/{style_def}]"

    def _collect_extras(self, record: logging.LogRecord) -> dict[str, object]:
        """Return LogRecord attributes as extras, filtered by prefix / allowlist."""

        extras: dict[str, object] = {}

        for k, v in record.__dict__.items():
            if k in _STANDARD_RECORD_KEYS:
                continue
            if k.startswith("_"):
                continue
            if self.extra_prefix and not k.startswith(self.extra_prefix):
                continue
            if self.extra_allowlist is not None and k not in self.extra_allowlist:
                continue
            extras[k] = v

        return extras

    def _truncate(self, s: str) -> str:
        if len(s) <= self.max_extra_len:
            return s

        return s[: self.max_extra_len - 1] + "…"

    def _format_extras_inline(self, extras: dict[str, object]) -> str:
        """Format extras into a single rich-markup inline string with explicit coloring."""

        if not extras:
            return ""

        key_style = "cyan"
        value_style = "green"
        secondary_style = "dim"

        parts: list[str] = []
        for k, v in extras.items():
            s = self._truncate(str(v))
            parts.append(
                f"[{key_style}]{k}[/{key_style}]"
                f"[{secondary_style}]=[/{secondary_style}]"
                f"[{value_style}]{s}[/{value_style}]"
            )

        return (
            "  "
            + f"[{secondary_style}]([/{secondary_style}]"
            + f"[{secondary_style}], [/{secondary_style}]".join(parts)
            + f"[{secondary_style}])[/{secondary_style}]"
        )

    def _format_extras_multiline(self, extras: dict[str, object]) -> str:
        """Pretty multiline extras block with a connector and bordered box."""

        if not extras:
            return ""

        items = list(extras.items())
        key_w = min(max(len(k) for k, _ in items), 24)

        rows: list[tuple[str, str]] = []
        max_row_len = 0
        for k, v in items:
            key = k.ljust(key_w)
            val = self._truncate(str(v))
            rows.append((key, val))
            max_row_len = max(max_row_len, len(f"{key} = {val}"))

        max_row_len = min(max_row_len, 80)

        key_style = "cyan"
        value_style = "green"
        secondary_style = "dim"

        label_name = "EXTRA"
        label_style = f"white on {BADGE_COLOR_MAP.get(label_name, DEFAULT_BADGE_COLOR)}"
        label = f"[{label_style}] {label_name} [/{label_style}]"

        indent = " " * 15

        pad_after_label = max(1, max_row_len - 6)
        top = (
            f"{indent}[{secondary_style}]└── ┌─[/]{label}[{secondary_style}]"
            + ("─" * pad_after_label)
            + f"┐[/{secondary_style}]"
        )

        lines: list[str] = [top]

        for key, val in rows:
            plain = f"{key} = {val}"
            pad = " " * max(0, max_row_len - len(plain))

            line = (
                f"{indent}    [{secondary_style}]│ [/{secondary_style}]"
                f"[{key_style}]{key}[/{key_style}]"
                f"[{secondary_style}] = [/{secondary_style}]"
                f"[{value_style}]{val}[/{value_style}]"
                f"{pad}"
                f"[{secondary_style}] │[/{secondary_style}]"
            )
            lines.append(line)

        bottom = (
            f"{indent}    [{secondary_style}]└"
            + ("─" * (max_row_len + 2))
            + f"┘[/{secondary_style}]"
        )
        lines.append(bottom)

        return "\n\n" + "\n".join(lines)

    def format(self, record: logging.LogRecord) -> str:
        """Format a LogRecord into a string."""

        message = record.getMessage()
        badge = self._format_level_badge(record)
        msg = f"  {badge}  {message}"

        if self.show_extra:
            extras = self._collect_extras(record)
            if extras:
                if self.extra_multiline:
                    msg += self._format_extras_multiline(extras)
                else:
                    msg += self._format_extras_inline(extras)

        return msg


class AutoCtxFilter(logging.Filter):
    """
    Automatically prefix extra fields.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        prefix = _EXTRA_PREFIX
        if not prefix:
            return True

        original = record.__dict__
        new_d: dict[str, object] = {}

        for key, value in original.items():
            # Keep standard LogRecord fields as-is
            if key in _STANDARD_RECORD_KEYS:
                new_d[key] = value
                continue

            # Keep private/internal attrs as-is
            if key.startswith("_"):
                new_d[key] = value
                continue

            # Rewrite only non-prefixed extras, preserving order
            if key.startswith(prefix):
                new_d[key] = value
            else:
                new_d[f"{prefix}{key}"] = value

        # Replace in-place
        original.clear()
        original.update(new_d)
        return True


def setup_logging(
    level: int | str = logging.INFO,
    name: str | None = "app",
    *,
    use_level_colors: bool = False,
    show_extra: bool = False,
    extra_allowlist: Iterable[str] | None = None,
    extra_prefix: str | None = "ctx_",
    extra_multiline: bool = False,
):
    """
    Configure a logger with StyledRichHandler and StyleFormatter.

    Args:
        level: Logging level constant or string.
        name: Logger name to configure. Use `None` for root logger.
        use_level_colors: If `True`, use different colors for each log level.
        show_extra: If `True`, append selected `extra` fields to log line.
        extra_allowlist: If provided, only show these extra keys.
        extra_prefix: If provided, only show extras whose key starts with this prefix.
        extra_multiline: If `True`, print extras as a pretty multi-line block.
    """

    global _EXTRA_PREFIX
    _EXTRA_PREFIX = extra_prefix

    handler = StyledRichHandler()
    handler.addFilter(AutoCtxFilter())

    allow = set(extra_allowlist) if extra_allowlist is not None else None

    handler.setFormatter(
        StyleFormatter(
            use_level_colors=use_level_colors,
            show_extra=show_extra,
            extra_allowlist=allow,
            extra_prefix=extra_prefix,
            extra_multiline=extra_multiline,
        )
    )

    if name is None:
        logging.basicConfig(level=level, handlers=[handler], force=True)
        return

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Replace handlers
    logger.handlers.clear()
    logger.addHandler(handler)

    # Child loggers propagate to this configured parent
    logger.propagate = False
