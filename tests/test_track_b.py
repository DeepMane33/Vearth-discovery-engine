"""Tests for Track B (solar-system) modules."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.track_b.adapter_base import DatasetAdapter
from engine.track_b.horizons import HorizonsAdapter
from engine.track_b.mars_rad import MarsRADAdapter


class _DummyAdapter(DatasetAdapter):
    """Concrete adapter for testing the base class."""

    def source_name(self) -> str:
        return "dummy"

    def get_cache_key(self, target, start_date, end_date):
        return f"dummy_{target}_{start_date}_{end_date}"

    def fetch(self, target, start_date, end_date):
        return {"raw": "data", "target": target}

    def parse(self, raw):
        return [{"record": 1, "source": raw["target"]}]

    def quality_flag(self, records):
        for r in records:
            r["quality_flag"] = "ok"
            r["quality_flags"] = []
        return records


def test_adapter_base_cache_roundtrip(tmp_path):
    """Cache save/load roundtrip works."""
    adapter = _DummyAdapter(cache_dir=str(tmp_path))
    key = "test_key_123"
    data = {"records": [{"a": 1}], "quality_summary": {"total": 1}}

    # Initially no cache
    assert adapter.load_cache(key) is None

    # Save and load
    adapter.save_cache(key, data)
    loaded = adapter.load_cache(key)
    assert loaded == data


def test_adapter_get_data_uses_cache(tmp_path):
    """get_data returns cached data without calling fetch."""
    adapter = _DummyAdapter(cache_dir=str(tmp_path))
    key = adapter.get_cache_key("Mars", "2026-01-01", "2026-01-02")
    adapter.save_cache(key, {"records": [{"cached": True}],
                             "quality_summary": {"total": 1},
                             "cached": True})

    result = adapter.get_data("Mars", "2026-01-01", "2026-01-02",
                              use_cache=True)
    assert result["cached"] is True
    assert result["records"][0]["cached"] is True


def test_adapter_quality_flag():
    """quality_flag adds ok/flags to records."""
    adapter = _DummyAdapter()
    records = [{"a": 1}, {"b": 2}]
    flagged = adapter.quality_flag(records)
    for r in flagged:
        assert r["quality_flag"] == "ok"
        assert r["quality_flags"] == []


def test_adapter_quality_summary():
    """_quality_summary computes correct counts."""
    adapter = _DummyAdapter()
    records = [
        {"quality_flag": "ok", "quality_flags": []},
        {"quality_flag": "parse_error", "quality_flags": ["parse_error"]},
        {"quality_flag": "ok", "quality_flags": []},
    ]
    summary = adapter._quality_summary(records)
    assert summary["total_records"] == 3
    assert summary["flagged_records"] == 1
    assert summary["clean_records"] == 2


def test_horizons_parse_kv_line():
    """_parse_kv_line handles both 'X = val' and 'VX=value' formats."""
    # Space-separated format
    vals = HorizonsAdapter._parse_kv_line(
        "X = 5.049113049487789E+07 Y =-2.083203224890075E+08 Z =-5.577837779914886E+06"
    )
    assert abs(vals["X"] - 5.049113049487789e7) < 1
    assert abs(vals["Y"] - -2.083203224890075e8) < 1e6
    assert abs(vals["Z"] - -5.577837779914886e6) < 1e3

    # No-space format
    vals2 = HorizonsAdapter._parse_kv_line(
        "VX= 2.445957741742463E+01 VY= 7.861473739133970E+00 VZ=-4.349787415986586E-01"
    )
    assert abs(vals2["VX"] - 24.45957741742463) < 0.001
    assert abs(vals2["VY"] - 7.86147373913397) < 0.001
    assert abs(vals2["VZ"] - -0.4349787415986586) < 0.001


def test_horizons_parse_full_record():
    """parse() correctly extracts a full ephemeris record."""
    adapter = HorizonsAdapter()
    raw_text = """$$SOE
2461041.500000000 = A.D. 2026-Jan-01 00:00:00.0000 TDB
 X = 5.049113049487789E+07 Y =-2.083203224890075E+08 Z =-5.577837779914886E+06
 VX= 2.445957741742463E+01 VY= 7.861473739133970E+00 VZ=-4.349787415986586E-01
 LT= 7.152428095847230E+02 RG= 2.144243999522301E+08 RR=-1.866796822438702E+00
2461042.500000000 = A.D. 2026-Jan-02 00:00:00.0000 TDB
 X = 5.260181527334687E+07 Y =-2.076305672451158E+08 Z =-5.615135105104402E+06
 VX= 2.439843101531509E+01 VY= 8.105035585732855E+00 VZ=-4.283755534011902E-01
 LT= 7.147067724698956E+02 RG= 2.142637000679967E+08 RR=-1.853071650710720E+00
$$EOE"""
    raw = {"raw_text": raw_text}
    records = adapter.parse(raw)
    assert len(records) == 2
    assert records[0]["datetime"] == "2026-Jan-01 00:00:00.0000 TDB"
    assert abs(records[0]["x_km"] - 5.049113049487789e7) < 1
    assert abs(records[0]["y_km"] - -2.083203224890075e8) < 1e6
    assert abs(records[0]["z_km"] - -5.577837779914886e6) < 1e3
    assert abs(records[0]["vx_km_s"] - 24.45957741742463) < 0.001
    assert abs(records[0]["range_km"] - 2.144243999522301e8) < 1e6
    assert records[1]["datetime"] == "2026-Jan-02 00:00:00.0000 TDB"


def test_horizons_parse_no_markers():
    """parse() returns empty list when $$SOE/$$EOE missing."""
    adapter = HorizonsAdapter()
    raw = {"raw_text": "No ephemeris markers here."}
    records = adapter.parse(raw)
    assert records == []


def test_horizons_quality_flag():
    """quality_flag marks missing/implausible values."""
    adapter = HorizonsAdapter()
    records = [
        {"x_km": 1.0e8, "y_km": 2.0e8, "z_km": 3.0e7,  # OK (realistic)
         "vx_km_s": 1.0e1, "vy_km_s": 2.0e1, "vz_km_s": 3.0e-1},
        {"x_km": None, "y_km": 2.0e8, "z_km": 3.0e7,  # Missing position
         "vx_km_s": 1.0e1, "vy_km_s": 2.0e1, "vz_km_s": 3.0e-1},
        {"x_km": 100.0, "y_km": 200.0, "z_km": 300.0,  # Implausibly close
         "vx_km_s": 1.0e1, "vy_km_s": 2.0e1, "vz_km_s": 3.0e-1},
    ]
    flagged = adapter.quality_flag(records)
    assert flagged[0]["quality_flag"] == "ok"
    assert "missing_position" in flagged[1]["quality_flags"]
    assert flagged[2]["quality_flag"] == "implausible_position"


def test_mars_rad_earth_date_to_sol():
    """_earth_date_to_sol converts Earth dates to MSL sols correctly."""
    adapter = MarsRADAdapter()
    # MSL landed 2012-08-06 (Sol 0)
    assert adapter._earth_date_to_sol("2012-08-06") == 0
    assert adapter._earth_date_to_sol("2012-08-07") == 1
    assert adapter._earth_date_to_sol("2012-08-05") == -1
    # One Mars year later (~687 Earth days)
    sol = adapter._earth_date_to_sol("2014-06-24")
    assert 680 < sol < 700


def test_mars_rad_find_sol_dirs():
    """_find_sol_directories filters directories by sol range."""
    adapter = MarsRADAdapter()
    # Format as HTML hrefs (as they appear in directory listing)
    dirs = [
        'href="SOL_00000_00089/"',
        'href="SOL_00090_00179/"',
        'href="SOL_00180_00269/"',
        'href="SOL_00270_00359/"',
        'href="SOL_00360_00449/"',
    ]
    matching = adapter._find_sol_directories(100, 200, dirs=dirs)
    assert "SOL_00090_00179" in matching
    assert "SOL_00180_00269" in matching
    assert "SOL_00000_00089" not in matching
    assert "SOL_00270_00359" not in matching


def test_mars_rad_parse_tabular():
    """_parse_rad_tabular extracts records from RAD data."""
    adapter = MarsRADAdapter()
    content = """/* PDS label header */
SOL   LMST   PRESSURE   TEMPERATURE   DOSE_RATE
1     12.5   750.0      -60.0         0.25
2     13.0   745.0      -62.0         0.23
"""
    records = adapter._parse_rad_tabular(content, "dummy.tab")
    assert len(records) == 2
    assert records[0]["sol"] == "1"
    assert records[0]["pressure"] == "750.0"
    assert records[1]["sol"] == "2"


def test_mars_rad_quality_flag():
    """quality_flag marks missing/implausible RAD values."""
    adapter = MarsRADAdapter()
    records = [
        {"sol": "1", "pressure": "750.0", "temperature": "-60.0",
         "dose_rate": "0.25"},  # OK
        {"sol": "2", "pressure": "", "temperature": "-60.0",
         "dose_rate": "0.23"},  # Missing pressure
        {"sol": "3", "pressure": "5000.0", "temperature": "-60.0",
         "dose_rate": "0.20"},  # Implausible pressure
    ]
    flagged = adapter.quality_flag(records)
    assert flagged[0]["quality_flag"] == "ok"
    assert "missing:pressure" in flagged[1]["quality_flags"]
    assert "implausible_pressure" in flagged[2]["quality_flags"]
