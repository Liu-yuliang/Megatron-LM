#!/usr/bin/env python3
"""OLMo3-backed ConceptLM V1 entrypoint.

This wrapper keeps the baseline GPT training path untouched and simply reuses
the ConceptLM V1 prototype with an OLMo3 backbone default.
"""

from pretrain_conceptlm_v1 import main as conceptlm_main


if __name__ == "__main__":
    conceptlm_main()
