"""
Stelline utilities module providing logger that wraps HOLOSCAN logging functionality.
"""

from stelline._logging import log_info, log_error, log_warn, log_debug


class HoloscanLogger:
    """
    Logger that wraps HOLOSCAN logging functions with Python logging interface.
    """

    def info(self, message: str) -> None:
        """Log info message using HOLOSCAN_LOG_INFO."""
        log_info(message)

    def warning(self, message: str) -> None:
        """Log warning message using HOLOSCAN_LOG_WARN."""
        log_warn(message)

    def warn(self, message: str) -> None:
        """Alias for warning()."""
        log_warn(message)

    def error(self, message: str) -> None:
        """Log error message using HOLOSCAN_LOG_ERROR."""
        log_error(message)

    def debug(self, message: str) -> None:
        """Log debug message using HOLOSCAN_LOG_DEBUG."""
        log_debug(message)


# Global logger instance
logger = HoloscanLogger()

