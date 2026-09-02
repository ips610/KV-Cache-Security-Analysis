"""Task-set loaders for this repo's experiments.

This file exists to make ``datasets`` a *regular* package rather than a
namespace package. Without it, an environment that also has the HuggingFace
``datasets`` library installed resolves the import to that library instead:
Python treats a directory with no ``__init__.py`` as a namespace portion, keeps
scanning ``sys.path``, and a regular package found anywhere later wins
outright -- so no amount of ``sys.path`` reordering makes the repo copy
authoritative.

Observed on the DGX (`datasets` present in the conda env), where every
``from datasets.<x>_dataset import ...`` in ``experiments/`` fails with
ModuleNotFoundError even though the file is right there on disk.

Nothing here uses the HuggingFace library; every ``datasets.*`` import in this
repo refers to these modules.
"""
