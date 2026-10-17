#!/usr/bin/env python
"""Download selected SmartSpaces cameras (+calibration, ground truth) and compact the ground truth.

    python scripts/perception_ingest.py --local-dir data/smartspaces --scene Warehouse_000 \
        --cameras Camera_0003 Camera_0010 Camera_0015 Camera_0001

Dataset nvidia/PhysicalAI-SmartSpaces is CC-BY-4.0 and NOT gated: no token is required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intercycle.perception.ingest import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
