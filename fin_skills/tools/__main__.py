"""`python -m fin_skills.tools` - print the tool definitions. See export.py for the flags."""
from __future__ import annotations

import sys

from fin_skills.tools.export import main

if __name__ == "__main__":                                      # pragma: no cover - CLI
    sys.exit(main())
