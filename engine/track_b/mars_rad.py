"""Mars RAD (Radiation Assessment Detector) adapter.

MSL/Curiosity RAD measures radiation on Mars' surface. Dataset:
MSL-M-RAD-3-RDR-V1.0 (PDS PPI Node - Planetary Plasma Interactions).

Data files are organized by sol ranges in subdirectories:
  https://pds-ppi.igpp.ucla.edu/data/MSL-M-RAD-3-RDR-V1.0/DATA/SOL_00000_00089/

Each sol directory contains .tab data files and .lbl label files.

PDS PPI Node: https://pds-ppi.igpp.ucla.edu/data/MSL-M-RAD-3-RDR-V1.0/DATA/
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from engine.track_b.pds_adapter import PDSDatasetAdapter


class MarsRADAdapter(PDSDatasetAdapter):
    """MSL Curiosity RAD radiation measurements from Mars surface."""

    # MSL RAD RDR data directory on PDS PPI Node
    DEFAULT_BASE_URL = (
        "https://pds-ppi.igpp.ucla.edu/data/MSL-M-RAD-3-RDR-V1.0/DATA/"
    )

    # MSL landing date: 2012-08-06 (Sol 0)
    MSL_LANDING = datetime(2012, 8, 6)

    def __init__(self, cache_dir: str | None = None,
                 base_url: str | None = None):
        super().__init__(
            dataset_id="MSL-M-RAD-3-RDR-V1.0",
            base_url=base_url or self.DEFAULT_BASE_URL,
            cache_dir=cache_dir,
            data_file_pattern=".tab",
        )
        self.min_interval = 2.0

    def source_name(self) -> str:
        return "PDS MSL RAD (Mars Radiation Assessment Detector)"

    def get_cache_key(self, target: str, start_date: str,
                      end_date: str) -> str:
        return f"pds_msl_rad_{target}_{start_date}_{end_date}"

    def fetch(self, target: str, start_date: str,
              end_date: str) -> dict:
        """Download RAD data for the given date range.

        Converts Earth dates to MSL sols, finds the corresponding sol-range
        directories on the PDS PPI node, and downloads .tab data files.
        """
        start_sol = self._earth_date_to_sol(start_date)
        end_sol = self._earth_date_to_sol(end_date)

        # Find sol-range directories covering the requested range
        sol_dirs = self._find_sol_directories(start_sol, end_sol)
        if not sol_dirs:
            raise RuntimeError(
                f"No PDS PPI sol directories found covering sols "
                f"{start_sol}-{end_sol} for MSL RAD"
            )

        downloaded = []
        for sol_dir in sol_dirs:
            dir_url = self.base_url.rstrip("/") + "/" + sol_dir + "/"
            try:
                self._rate_limit()
                r = requests_get(dir_url)
                r.raise_for_status()
                files = self._list_data_files(r.text)
                for fname in files:
                    file_url = dir_url + fname
                    try:
                        self._rate_limit()
                        fr = requests_get(file_url, timeout=60)
                        fr.raise_for_status()
                        downloaded.append({
                            "filename": fname,
                            "url": file_url,
                            "content": fr.text,
                            "size_bytes": len(fr.content),
                            "sol_dir": sol_dir,
                        })
                    except Exception:
                        pass
            except Exception:
                pass

        if not downloaded:
            raise RuntimeError(
                f"Failed to download any RAD data files from sol directories"
            )

        return {
            "dataset_id": self.dataset_id,
            "base_url": self.base_url,
            "files": downloaded,
            "target": target,
            "start_date": start_date,
            "end_date": end_date,
            "start_sol": start_sol,
            "end_sol": end_sol,
        }

    def _earth_date_to_sol(self, date_str: str) -> int:
        """Convert an Earth date string to MSL sol number."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        delta = dt - self.MSL_LANDING
        return delta.days

    def _find_sol_directories(self, start_sol: int,
                              end_sol: int,
                              dirs: list[str] | None = None) -> list[str]:
        """Query the PDS PPI base directory and find sol-range dirs covering
        the requested sol range.

        If dirs is provided, use it instead of fetching from the network
        (useful for testing).
        """
        if dirs is None:
            # List the base DATA directory
            try:
                r = requests_get(self.base_url, timeout=30)
                r.raise_for_status()
            except Exception:
                return []
            html = r.text
        else:
            html = "\n".join(dirs)

        # Parse directory listing for SOL_XXXXX_XXXXX/ directories
        all_dirs = []
        for line in html.split("\n"):
            m = re.search(r'href="(SOL_\d+_\d+)/"', line)
            if m:
                all_dirs.append(m.group(1))

        # Filter to directories overlapping the requested range
        matching = []
        for d in sorted(all_dirs):
            parts = d.replace("SOL_", "").replace("/", "").split("_")
            if len(parts) == 2:
                try:
                    d_start = int(parts[0])
                    d_end = int(parts[1])
                    if d_end >= start_sol and d_start <= end_sol:
                        matching.append(d)
                except ValueError:
                    continue

        return matching

    def parse(self, raw: dict) -> list[dict]:
        """Parse RAD data files into measurement records."""
        records = []
        for file_info in raw.get("files", []):
            content = file_info.get("content", "")
            fname = file_info.get("filename", "")

            if fname.endswith(".lbl"):
                records.append({
                    "filename": fname,
                    "file_type": "label",
                    "content_preview": content[:300],
                    "quality_flag": "metadata",
                    "sol_dir": file_info.get("sol_dir", ""),
                })
                continue

            file_records = self._parse_rad_tabular(content, fname)
            for rec in file_records:
                rec["sol_dir"] = file_info.get("sol_dir", "")
            records.extend(file_records)

        return records

    def _parse_rad_tabular(self, content: str, fname: str) -> list[dict]:
        """Parse RAD tabular data (space/tab delimited with headers)."""
        records = []
        lines = content.split("\n")

        header_idx = None
        headers = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("/*"):
                if any(k in stripped.lower() for k in
                       ("sol", "time", "pressure", "temp", "dose", "radiation")):
                    headers = stripped.split()
                    header_idx = i
                    break

        if headers is None:
            return [{
                "filename": fname,
                "error": "No header found in RAD data file",
                "quality_flag": "parse_error",
            }]

        for line in lines[header_idx + 1:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("/*"):
                continue
            parts = stripped.split()
            if len(parts) < len(headers):
                continue
            try:
                record = {}
                for h, v in zip(headers, parts):
                    record[h.lower()] = v
                record["_source_file"] = fname
                record["_source_dataset"] = "MSL-M-RAD-3-RDR-V1.0"
                records.append(record)
            except Exception:
                continue

        return records

    def quality_flag(self, records: list[dict]) -> list[dict]:
        """Flag RAD records with missing or implausible values."""
        for r in records:
            flags = []
            if r.get("quality_flag") == "metadata":
                continue

            if r.get("error"):
                flags.append("parse_error")
                r["quality_flag"] = "parse_error"
                r["quality_flags"] = flags
                continue

            for key in ("sol", "pressure", "temperature", "dose_rate"):
                val = r.get(key)
                if val in (None, "", "nan", "NaN", "-999", "-999.0"):
                    flags.append(f"missing:{key}")

            try:
                pres = float(r.get("pressure", 0))
                if pres < 0 or pres > 2000:
                    flags.append("implausible_pressure")
            except (ValueError, TypeError):
                flags.append("non_numeric_pressure")

            try:
                temp = float(r.get("temperature", 0))
                if temp < -150 or temp > 50:
                    flags.append("implausible_temperature")
            except (ValueError, TypeError):
                flags.append("non_numeric_temperature")

            r["quality_flag"] = flags[0] if flags else "ok"
            r["quality_flags"] = flags

        return records


def requests_get(url, timeout=30):
    """Wrapper around requests.get to allow easy mocking in tests."""
    import requests
    return requests.get(url, timeout=timeout)
