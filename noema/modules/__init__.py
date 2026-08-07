"""Noema Modules — pluggable domain modules that work standalone and together."""

from .registry import ModuleRegistry, NoemaModule, get_registry

__all__ = ["ModuleRegistry", "NoemaModule", "get_registry"]
