"""SRG Lite core: shared pipeline and evaluation harness."""

from .defaults import LITE_DISCOVERY_OPTIONS, discovery_options_with_top_k
from .pipeline import generate_reading_list

__all__ = [
    "LITE_DISCOVERY_OPTIONS",
    "discovery_options_with_top_k",
    "generate_reading_list",
]
