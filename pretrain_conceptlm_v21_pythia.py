#!/usr/bin/env python3
# Copyright (c) 2026, ConceptLM contributors.

"""Pythia-backed ConceptLM V2.1 entrypoint."""

from __future__ import annotations

import sys

from pretrain_conceptlm_v21 import main as conceptlm_v21_main


def main() -> None:
    if not any(
        arg == "--conceptlm-backbone" or arg.startswith("--conceptlm-backbone=")
        for arg in sys.argv[1:]
    ):
        sys.argv.extend(["--conceptlm-backbone", "pythia"])
    conceptlm_v21_main()


if __name__ == "__main__":
    main()
