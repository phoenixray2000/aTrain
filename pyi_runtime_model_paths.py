from pathlib import Path
import sys

import aTrain_core.globals as core_globals


if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    package_dir = Path(sys._MEIPASS) / "aTrain"
    core_globals.MODELS_DIR = package_dir / "models"
    core_globals.REQUIRED_MODELS_DIR = package_dir / "required_models"
