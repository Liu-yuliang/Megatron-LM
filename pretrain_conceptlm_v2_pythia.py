#!/usr/bin/env python3
# Copyright (c) 2026, ConceptLM contributors.

"""Pythia-backed ConceptLM V2 entrypoint."""

import sys

from pretrain_conceptlm_v2 import main as conceptlm_v2_main


def _ensure_pythia_backbone_default() -> None:
    has_backbone_arg = any(
        arg == "--conceptlm-v2-backbone" or arg.startswith("--conceptlm-v2-backbone=")
        for arg in sys.argv[1:]
    )
    if not has_backbone_arg:
        sys.argv.extend(["--conceptlm-v2-backbone", "pythia"])


if __name__ == "__main__":
    _ensure_pythia_backbone_default()
    conceptlm_v2_main()
