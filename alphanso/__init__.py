"""Public ALPHANSO package API."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__.split(".", 1)[0])
except PackageNotFoundError:  # Source tree before installation.
    __version__ = "development"

from .transport import Transport
from .data_manager import ensure_data, get_data_dir, is_data_available, DATA_VERSION

__all__ = ['Transport', '__version__', 'ensure_data', 'get_data_dir',
           'is_data_available', 'DATA_VERSION']
