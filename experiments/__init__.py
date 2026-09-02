"""Regular package marker.

Without this file ``experiments`` would be a namespace package and, once the
upstream KVCOMM checkout is on ``sys.path``, Python would merge its
``experiments/`` directory (the original MMLU/GSM8K/HumanEval runners) into
this one. Keeping it a regular package pins ``experiments.*`` to this repo.
"""
