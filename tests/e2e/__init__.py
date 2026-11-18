"""E2E Test Suite for Autonomous Telegram Userbot.

Covers Tiers 1-4:
- Tier 1: Feature Coverage (Features 1-10)
- Tier 2: Boundary & Corner Cases (Features 1-10)
- Tier 3: Cross-Feature Interactions & Pairwise Coverage
- Tier 4: Real-World Application Scenarios
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
