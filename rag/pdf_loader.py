"""Backward-compatible PDF loader wrapper.

New code should import utils.pdf_loader.load_pdf directly.
"""

from utils.pdf_loader import load_pdf


__all__ = ["load_pdf"]
