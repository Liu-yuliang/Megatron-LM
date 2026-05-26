#!/usr/bin/env python3
"""Pythia-backed ConceptLM V1 entrypoint.

This wrapper keeps the baseline Pythia training path untouched and simply
selects the ConceptLM V1 prototype with a GPT-NeoX/Pythia backbone default.
"""

import sys

from pretrain_conceptlm_v1 import main as conceptlm_main


def _ensure_pythia_backbone_default() -> None:
    has_backbone_arg = any(
        arg == "--conceptlm-backbone" or arg.startswith("--conceptlm-backbone=")
        for arg in sys.argv[1:]
    )
    if not has_backbone_arg:
        sys.argv.extend(["--conceptlm-backbone", "pythia"])


if __name__ == "__main__":
    _ensure_pythia_backbone_default()
    conceptlm_main()
