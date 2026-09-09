"""`python -m fin_skills.mcp` - serve the guards over MCP stdio. See __init__ for flags."""
from __future__ import annotations

import sys

from fin_skills.mcp import main

if __name__ == "__main__":                                      # pragma: no cover - CLI
    sys.exit(main())
