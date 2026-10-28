#!/usr/bin/env python3
"""Generate all research engine modules."""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(BASE, "engine")

def w(name, content):
    path = os.path.join(ENGINE, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  OK: {name}")

print("Building VEARTH research modules...")
