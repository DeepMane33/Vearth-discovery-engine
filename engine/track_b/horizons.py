"""JPL Horizons ephemeris adapter.

Fetches ephemerides (position, velocity, physical parameters) for solar
system bodies from JPL Horizons API.

Verified working: https://ssd.jpl.nasa.gov/api/horizons.api returns 200
for Mars (COMMAND=499) vector ephemeris.

Rate limit: JPL asks for no more than 1 request per second.
"""

from __future__ import annotations

import requests
from engine.track_b.adapter_base import DatasetAdapter


class HorizonsAdapter(DatasetAdapter):
    """JPL Horizons ephemerides for solar system bodies."""

    def __init__(self, cache_dir: str | None = None):
        super().__init__(cache_dir)
        self.min_interval = 1.0
        self.base_url = "https://ssd.jpl.nasa.gov/api/horizons.api"

    def source_name(self) -> str:
        return "JPL Horizons"

    def get_cache_key(self, target: str, start_date: str,
                      end_date: str) -> str:
        return f"horizons_{target}_{start_date}_{end_date}"

    def fetch(self, target: str, start_date: str,
              end_date: str) -> dict:
        """Query Horizons for vector ephemeris of target.

        target: Horizons COMMAND value (e.g. "499" for Mars, "301" for Moon,
                "599" for Jupiter, "699" for Saturn, or name like "Mars").
        """
        params = {
            "format": "json",
            "COMMAND": target,
            "OBJ_DATA": "YES",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "VECTORS",
            "CENTER": "500@0",  # Sun
            "START_TIME": start_date,
            "STOP_TIME": end_date,
            "STEP_SIZE": "1d",
        }

        self._rate_limit()
        r = requests.get(self.base_url, params=params, timeout=30)
        r.raise_for_status()

        data = r.json()
        if data.get("result") is None:
            raise RuntimeError(
                f"Horizons returned no result for '{target}' "
                f"({start_date} to {end_date})"
            )

        return {
            "raw_text": data["result"],
            "signature": data.get("signature", ""),
            "target": target,
            "start_date": start_date,
            "end_date": end_date,
        }

    def parse(self, raw: dict) -> list[dict]:
        """Parse Horizons vector ephemeris text into records.

        Horizons vector output format (between $$SOE and $$EOE):
          JD = calendar date
           X = ... Y = ... Z = ...   (km)
          VX= ... VY= ... VZ= ...   (km/s)
          LT= ... RG= ... RR= ...

        Each record: datetime, x, y, z (km), vx, vy, vz (km/s),
        and light-time / range / range-rate.
        """
        text = raw["raw_text"]
        lines = text.split("\n")

        # Find the $$SOE (start of ephemeris) marker
        soe_idx = None
        eoe_idx = None
        for i, line in enumerate(lines):
            if "$$SOE" in line:
                soe_idx = i
            if "$$EOE" in line:
                eoe_idx = i

        if soe_idx is None or eoe_idx is None:
            return []

        records = []
        i = soe_idx + 1
        while i < eoe_idx:
            line = lines[i].strip()
            # Each record starts with a JD line: "2461041.500000000 = A.D. 2026-Jan-01 ..."
            if "=" in line and "A.D." in line:
                # Parse datetime from the JD line
                parts = line.split("=", 1)
                if len(parts) == 2:
                    date_str = parts[1].strip()
                    # date_str like: "A.D. 2026-Jan-01 00:00:00.0000 TDB "
                    date_parts = date_str.split()
                    if len(date_parts) >= 4:
                        # "A.D. 2026-Jan-01 00:00:00.0000 TDB"
                        datetime_str = " ".join(date_parts[1:4])
                    elif len(date_parts) >= 2:
                        datetime_str = " ".join(date_parts[1:3])
                    else:
                        datetime_str = date_str
                else:
                    datetime_str = line.strip()

                # Next line: X = ... Y = ... Z = ...
                pos_line = ""
                # Next line: VX= ... VY= ... VZ= ...
                vel_line = ""
                # Next line: LT= ... RG= ... RR= ...
                misc_line = ""

                j = i + 1
                while j < eoe_idx:
                    lj = lines[j].strip()
                    if lj.startswith("X ") or lj.startswith("X="):
                        pos_line = lj
                    elif lj.startswith("VX"):
                        vel_line = lj
                    elif lj.startswith("LT"):
                        misc_line = lj
                        break
                    j += 1

                # Parse position
                pos_vals = self._parse_kv_line(pos_line)
                # Parse velocity
                vel_vals = self._parse_kv_line(vel_line)
                # Parse misc
                misc_vals = self._parse_kv_line(misc_line)

                record = {
                    "datetime": datetime_str,
                    "x_km": pos_vals.get("X"),
                    "y_km": pos_vals.get("Y"),
                    "z_km": pos_vals.get("Z"),
                    "vx_km_s": vel_vals.get("VX"),
                    "vy_km_s": vel_vals.get("VY"),
                    "vz_km_s": vel_vals.get("VZ"),
                    "lt_sec": misc_vals.get("LT"),
                    "range_km": misc_vals.get("RG"),
                    "range_rate_km_s": misc_vals.get("RR"),
                }
                records.append(record)
                i = j + 1
            else:
                i += 1

        return records

    @staticmethod
    def _parse_kv_line(line: str) -> dict:
        """Parse a Horizons key=value line.

        Handles both 'X = 1.23E+07 Y =-4.56E+07' and
        'VX= 2.44E+01 VY=7.86E+00 VZ=-4.35E-01' formats.
        """
        import re
        vals = {}
        if not line:
            return vals
        # Match KEY = VALUE or KEY=VALUE patterns
        # KEY is alphanumeric, VALUE is a number (possibly scientific notation)
        pattern = r"([A-Za-z]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        for match in re.finditer(pattern, line):
            key = match.group(1).strip()
            val_str = match.group(2).strip()
            try:
                vals[key] = float(val_str)
            except (ValueError, TypeError):
                vals[key] = val_str
        return vals


    def quality_flag(self, records: list[dict]) -> list[dict]:
        """Flag records with missing or implausible values."""
        for r in records:
            flags = []
            if r.get("x_km") is None or r.get("y_km") is None or r.get("z_km") is None:
                flags.append("missing_position")
            if r.get("vx_km_s") is None or r.get("vy_km_s") is None or r.get("vz_km_s") is None:
                flags.append("missing_velocity")
            # Position should be non-zero for a real ephemeris
            if all(r.get(k) is not None for k in ("x_km", "y_km", "z_km")):
                r_sq = r["x_km"] ** 2 + r["y_km"] ** 2 + r["z_km"] ** 2
                if r_sq < 1e6:  # less than 1000 km from Sun center
                    flags.append("implausible_position")
            r["quality_flag"] = flags[0] if flags else "ok"
            r["quality_flags"] = flags
        return records
