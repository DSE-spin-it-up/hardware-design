"""Top-level pipeline package: convergence loop, reporting, plots, config.

Avoid importing heavy submodules at package import time to prevent circular
imports when other modules import subpackages such as ``pipeline.helpers``.
Import the specific submodules or objects where they are needed instead.
"""

__all__ = ["config", "helpers", "loop", "plots", "reporting"]
