"""Lightweight service contracts; importing this package never loads model libraries."""

from .catalog import Capability, load_catalog

__all__ = ["Capability", "load_catalog"]
