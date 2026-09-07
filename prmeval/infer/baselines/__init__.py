"""Built-in inference baselines, imported only when selected."""

from __future__ import annotations

from importlib import import_module

# Map the public registry name from InferConfig to the module whose import-time
# @register_infer decorator adds the implementation to the INFERS registry.
# Multiple names may intentionally share one implementation module.
_INFER_MODULES = {
    "openai_compatible": ".openai_compatible_model",
    "rewind": ".rbm_model",
    "robodopamine": ".robodopamine",
    "robometer": ".rbm_model",
    "roboreward": ".roboreward",
    "sole_r1": ".sole_r1_model",
    "topreward": ".topreward",
}

# Map lazily exposed class attributes to their defining modules.  This keeps
# imports such as ``from ...baselines import RemoteModel`` compatible without
# eagerly importing every baseline and its optional heavyweight dependencies.
_CLASS_MODULES = {
    "RBMModel": ".rbm_model",
    "RemoteModel": ".openai_compatible_model",
    "RoboDopamine": ".robodopamine",
    "RoboReward": ".roboreward",
    "SoleR1": ".sole_r1_model",
    "TopReward": ".topreward",
}


def builtin_infer_names() -> list[str]:
    """Return built-in registry names without importing their modules."""
    return sorted(_INFER_MODULES)


def load_builtin_infer(name: str) -> None:
    """Import the selected implementation so its decorator registers it."""
    module = _INFER_MODULES.get(name.strip().lower())
    if module is not None:
        # A leading dot makes this relative to this package (``__name__``).
        # Python caches the result in sys.modules, so later calls return the
        # same module without executing its registration code again.
        import_module(module, __name__)


def __getattr__(name: str):
    """Lazily resolve classes exported by this package (PEP 562)."""
    module = _CLASS_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Import the defining module only when the class attribute is requested.
    return getattr(import_module(module, __name__), name)


__all__ = [
    "RBMModel",
    "RemoteModel",
    "RoboDopamine",
    "RoboReward",
    "SoleR1",
    "TopReward",
]
