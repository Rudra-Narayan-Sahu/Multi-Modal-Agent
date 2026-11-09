import os
import sys
from pathlib import Path

# Add project root to sys.path so 'app' package is always found
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Remove any shadowed 'app' module
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

# Import and execute dashboard
import app.ui.dashboard
