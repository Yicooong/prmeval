"""Stage 2 inference interfaces.

Built-in implementations are imported on demand by the evaluator.  Importing this
package must stay lightweight so that CLI help does not initialize local model stacks.
"""

from .base import Infer, RemoteError

__all__ = [
    "Infer",
    "RemoteError",
]
