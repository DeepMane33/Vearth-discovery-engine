"""PDS (Planetary Data System) dataset adapter framework.

Downloads planetary science data from PDS archive nodes. Supports direct
URL-based dataset access when the PDS search API endpoint is not directly
queriable.

PDS nodes that host solar system data:
  - Geosciences (pds-geosciences.wustl.edu): Mars, Moon, Mercury, etc.
  - Atmospheres (pds-atmospheres.nmsu.edu): Mars, Venus, Earth atmospheres
  - Small Bodies (sbn.psi.edu): asteroids, comets
  - Ring-Moon Systems (pds-rings.seti.org): rings, moons
  - Planetary Plasma Interactions (pds-ppi.igpp.ucla.edu): magnetometer

Rate limits: be respectful, cache everything.
"""

from __future__ import annotations

import csv
import io
import os
import time
from pathlib import Path
from typing import Any

import requests

from engine.track_b.adapter_base import DatasetAdapter


class PDSDatasetAdapter(DatasetAdapter):
    """Base PDS dataset adapter for direct file downloads."""

    def __init__(self, dataset_id: str, base_url: str,
                 cache_dir: str | None = None,
                 data_file_pattern: str | None = None):
        """
        Args:
            dataset_id: PDS dataset ID (e.g. "MSL-M-RAD-2-RDR-V1.0")
            base_url: base URL for the dataset directory on the PDS node
            data_file_pattern: filename pattern to match data files
        """
        super().__init__(cache_dir)
        self.dataset_id = dataset_id
        self.base_url = base_url
        self.data_file_pattern = data_file_pattern
        self.min_interval = 2.0  # PDS nodes are more sensitive

    def source_name(self) -> str:
        return f"PDS ({self.dataset_id})"

    def get_cache_key(self, target: str, start_date: str,
                      end_date: str) -> str:
        return f"pds_{self.dataset_id}_{target}_{start_date}_{end_date}"

    def fetch(self, target: str, start_date: str,
              end_date: str) -> dict:
        """Download data files from PDS dataset directory.

        Lists the directory, finds matching data files, downloads them,
        and returns raw content.
        """
        self._rate_limit()
        r = requests.get(self.base_url, timeout=30)
        r.raise_for_status()

        # Parse directory listing for data files
        files = self._list_data_files(r.text)
        if not files:
            raise RuntimeError(
                f"No data files found in {self.base_url} "
                f"(pattern: {self.data_file_pattern})"
            )

        downloaded = []
        for fname in files[:5]:  # limit to first 5 files for safety
            file_url = self.base_url.rstrip("/") + "/" + fname
            try:
                self._rate_limit()
                fr = requests.get(file_url, timeout=60)
                fr.raise_for_status()
                downloaded.append({
                    "filename": fname,
                    "url": file_url,
                    "content": fr.text,
                    "size_bytes": len(fr.content),
                })
            except Exception as e:
                # Skip files that fail, don't crash
                pass

        if not downloaded:
            raise RuntimeError(
                f"Failed to download any data files from {self.base_url}"
            )

        return {
            "dataset_id": self.dataset_id,
            "base_url": self.base_url,
            "files": downloaded,
            "target": target,
            "start_date": start_date,
            "end_date": end_date,
        }

    def _list_data_files(self, html: str) -> list[str]:
        """Extract data file links from PDS directory listing HTML."""
        files = []
        for line in html.split("\n"):
            # Look for .tab, .csv, .lbl, .dat, .img extensions
            for ext in (".tab", ".csv", ".lbl", ".dat", ".img", ".fit"):
                idx = line.find(ext)
                if idx > 0:
                    # Extract filename (look backward for href or quote)
                    start = max(0, idx - 100)
                    segment = line[start:idx + len(ext)]
                    # Find the actual filename
                    for sep in ('"', "'", ">", "="):
                        parts = segment.split(sep)
                        candidate = parts[-1].strip()
                        if candidate.endswith(ext) and len(candidate) < 200:
                            if candidate not in files:
                                files.append(candidate)
                            break
        return files

    def parse(self, raw: dict) -> list[dict]:
        """Parse downloaded PDS data files into records.

        Supports tab-delimited (.tab) and CSV (.csv) files with PDS labels.
        """
        records = []
        for file_info in raw.get("files", []):
            content = file_info.get("content", "")
            fname = file_info.get("filename", "")

            if fname.endswith(".csv"):
                records.extend(self._parse_csv(content, fname))
            elif fname.endswith(".tab"):
                records.extend(self._parse_tab(content, fname))
            else:
                # Non-tabular file, store as metadata record
                records.append({
                    "filename": fname,
                    "content_type": "raw_text",
                    "content_preview": content[:500],
                    "size_bytes": file_info.get("size_bytes", 0),
                })

        return records

    def _parse_csv(self, content: str, fname: str) -> list[dict]:
        records = []
        try:
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                row["_source_file"] = fname
                records.append(row)
        except Exception as e:
            records.append({
                "filename": fname,
                "error": f"CSV parse failed: {e}",
                "quality_flag": "parse_error",
            })
        return records

    def _parse_tab(self, content: str, fname: str) -> list[dict]:
        records = []
        try:
            reader = csv.DictReader(io.StringIO(content), delimiter="\t")
            for row in reader:
                row["_source_file"] = fname
                records.append(row)
        except Exception as e:
            records.append({
                "filename": fname,
                "error": f"TAB parse failed: {e}",
                "quality_flag": "parse_error",
            })
        return records

    def quality_flag(self, records: list[dict]) -> list[dict]:
        """Flag records with missing values or parse errors."""
        for r in records:
            flags = []
            if r.get("error"):
                flags.append("parse_error")
            # Check for missing critical values
            critical_keys = [k for k in r.keys()
                             if k not in ("_source_file", "error",
                                          "content_type", "content_preview",
                                          "size_bytes", "quality_flag",
                                          "quality_flags")]
            missing = [k for k in critical_keys
                       if r.get(k) in (None, "", "null", "NAN", "nan")]
            if missing:
                flags.append(f"missing:{','.join(missing[:3])}")
            r["quality_flag"] = flags[0] if flags else "ok"
            r["quality_flags"] = flags
        return records
