"""twin/site.yaml (CAL v1, lane S2): schema-valid, train-only provenance, and the CAL v1 invariants."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from intercycle import esk
from twin import provenance as PV
from twin.schemas import site as SS
from twin.schemas.prereg import parse_split

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "twin" / "site.yaml"
CAL = ROOT / "twin" / "cal"


@pytest.fixture(scope="module")
def site() -> SS.Site:
    import yaml

    return SS.Site.model_validate(yaml.safe_load(SITE.read_text(encoding="utf-8")))


def test_site_sources_are_train_only(site):
    rows = parse_split(esk.active_split_path(ROOT).read_text(encoding="utf-8"))
    assert PV.assert_train_only(PV.source_clips(site=site), rows) > 0


def test_every_field_is_tagged_and_cal_is_v1(site):
    for f in site.framings:
        assert f.H_img_to_floor.source == "cal_v1" and f.k1.source == "cal_v1"
        assert f.H_img_to_floor.note and "not by a human" in f.H_img_to_floor.note
    for p in site.presses:
        assert p.dilation_m.source == "cal_v1" and p.dilation_m.value > 0
    assert {z.id: z.framings for z in site.zones} == {"z9": ["c9_f0", "c9_f1"], "z14": ["c14_f0"]}


def test_scale_interval_contains_prior(site):
    s = site.scale.value
    assert s.lo <= 1.0 == s.mid <= s.hi


def test_presses_off_the_visible_floor_and_zones_disjoint(site):
    z = {zz.id: np.asarray(zz.floor_polygon.value) for zz in site.zones}
    assert z["z9"][:, 0].max() < z["z14"][:, 0].min()
    for p in site.presses:
        c = np.asarray(p.footprint.value).mean(0)
        assert not SS.point_in_polygon(float(c[0]), float(c[1]), [tuple(v) for v in z[p.zone]]), p.id


def test_v0_cal_files_marked_invalid():
    for name in ("k1_v0.json", "scale_v0.json"):
        assert json.loads((CAL / name).read_text(encoding="utf-8"))["_status"].startswith("INVALID")


def test_k1_v1_sources_are_train_and_per_framing():
    k1 = json.loads((CAL / "k1_v1.json").read_text(encoding="utf-8"))
    rows = parse_split(esk.active_split_path(ROOT).read_text(encoding="utf-8"))
    for f in ("c9_f0", "c9_f1", "c14_f0"):
        assert k1[f]["k1"] < 0
        assert PV.assert_train_only({f: k1[f]["source_clip_ids"]}, rows) > 0
