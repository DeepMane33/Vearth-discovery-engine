#!/usr/bin/env python3
"""Build script - generates all new research engine modules."""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(BASE, "engine")

os.makedirs(os.path.join(BASE, "scripts"), exist_ok=True)

# We'll build files one at a time
def write_module(filename, content):
    path = os.path.join(ENGINE, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Written: {filename} ({len(content)} bytes)")

print("Building VEARTH research engine modules...")

# ============================================================
# Import the builder functions from separate build files
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_paper_ingestion import get_content as c1
from build_methodology_extractor import get_content as c2
from build_research_knowledge import get_content as c3
from build_gap_detector import get_content as c4
from build_research_orchestrator import get_content as c5

write_module("paper_ingestion.py", c1())
write_module("methodology_extractor.py", c2())
write_module("research_knowledge.py", c3())
write_module("gap_detector.py", c4())
write_module("research_orchestrator.py", c5())

print("\nDone! All modules built.")
