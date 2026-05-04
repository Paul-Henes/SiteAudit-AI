from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_DB_PATH = ROOT / ".tmp" / "test-siteaudit.db"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
os.environ["SITEAUDIT_DB_PATH"] = str(TEST_DB_PATH)
