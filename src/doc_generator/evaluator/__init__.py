"""Documentation evaluation and comparison module."""

from .downloader import DocumentationDownloader
from .comparator import DocumentationComparator
from .metrics import SimilarityMetrics

__all__ = [
    'DocumentationDownloader',
    'DocumentationComparator',
    'SimilarityMetrics'
]