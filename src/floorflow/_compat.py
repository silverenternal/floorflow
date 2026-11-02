"""Lazy aliases for imports that predate the feature-package layout."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

from floorflow._module_map import MODULE_MAP


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module) -> None:
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        prefix = "floorflow."
        if not fullname.startswith(prefix):
            return None
        short_name = fullname[len(prefix):]
        mapped = MODULE_MAP.get(short_name)
        if mapped is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(mapped))


def install_legacy_aliases() -> None:
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.append(_AliasFinder())


__all__ = ["install_legacy_aliases"]
