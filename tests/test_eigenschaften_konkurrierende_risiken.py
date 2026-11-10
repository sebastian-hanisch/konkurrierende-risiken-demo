"""Eigenschaftstests (metamorph, DEMO-PLAYBOOK §12 / ORACLE-PLAYBOOK §5.2): Aussagen, die für jeden Datensatz gelten müssen, ohne dass man das richtige Ergebnis kennt.
Je Eigenschaft 60 Zufallsdatensätze mit festem Seed (ganze Tage, also mit Gleichständen) plus Randfälle.

  * Summe: CIF LKW + CIF Bahn + noch da = 1 an jedem Punkt.
  * Monotonie und Wertebereich: jede kumulierte Inzidenz wächst, bleibt in [0, 1] und liegt unter 1 − Anteil, der noch da ist.
  * Ohne Konkurrenz (kein Bahn-Ereignis vor der letzten LKW-Abholung): CIF = 1 − Kaplan–Meier.
  * Naiv ≥ kumuliert für jeden Datensatz (Beweis: Produkt der Faktoren 1 − d1/n liegt über dem Produkt 1 − (d1 + d2)/n, siehe 📐 in der App).
  * Invarianzen: Umsortieren der Container, Skalieren der Zeitachse, Vertauschen der Wege."""

import numpy as np
import pytest

import kkr_core as K
import kkr_model as M
from kkr_scenario import BAHN, LKW, Sample


def datasets(count=60, seed=7, n_max=30):
    rng = K.SplitMix64(seed)
    for _ in range(count):
        n = 3 + int(rng.uniform(1)[0] * (n_max - 2))
        t = np.ceil(rng.uniform(n) * 10)
        c = np.floor(rng.uniform(n) * 3).astype(int)
        yield Sample(t, c)


GRID = np.arange(0.5, 12.0, 0.5)


def test_summe_der_anteile_ist_ueberall_eins():
    for s in datasets():
        c = M.curves(s, GRID)
        assert c.sum_correct == pytest.approx(np.ones(GRID.size), abs=1e-12)


def test_kumulierte_inzidenz_waechst_bleibt_im_bereich_und_liegt_unter_eins_minus_rest():
    for s in datasets():
        c = M.curves(s, GRID)
        for cif in (c.cif_lkw, c.cif_bahn):
            assert np.all(np.diff(cif) >= -1e-15) and np.all(cif >= 0) and np.all(cif <= 1 + 1e-12)
            assert np.all(cif <= 1 - c.remaining + 1e-12)
        assert np.all(np.diff(c.remaining) <= 1e-15)


def test_ohne_bahn_ist_die_kumulierte_inzidenz_eins_minus_kaplan_meier():
    for s in datasets():
        ohne = Sample(s.time, np.where(s.cause == BAHN, 0, s.cause))     # alle Bahn-Abholungen zu Zensierungen gemacht: keine Konkurrenz
        c = M.curves(ohne, GRID)
        assert c.cif_lkw == pytest.approx(c.naive_lkw, abs=1e-12)
        assert np.all(c.cif_bahn == 0)


def test_naive_rechnung_liegt_nie_unter_der_kumulierten_inzidenz():
    for s in datasets():
        c = M.curves(s, GRID)
        assert np.all(c.naive_lkw >= c.cif_lkw - 1e-12) and np.all(c.naive_bahn >= c.cif_bahn - 1e-12)


def test_gleich_sind_beide_genau_dann_wenn_die_bahn_vor_der_letzten_lkw_abholung_nichts_abholt():
    for s in datasets(count=120, seed=21, n_max=8):
        lkw_times = s.time[s.cause == LKW]
        if lkw_times.size == 0:
            continue
        last = lkw_times.max()
        bahn_before = bool(np.any((s.cause == BAHN) & (s.time < last)))
        e = M.estimate_at(s, float(last))
        assert (abs(e.naive_lkw - e.cif_lkw) > 1e-12) == bahn_before, (s.time, s.cause)


def test_umsortieren_der_container_aendert_nichts():
    rng = K.SplitMix64(5)
    for s in datasets(count=30):
        perm = np.argsort(rng.uniform(s.time.size))
        a, b = M.curves(s, GRID), M.curves(Sample(s.time[perm], s.cause[perm]), GRID)
        for f in ("cif_lkw", "cif_bahn", "remaining", "naive_lkw", "naive_bahn"):
            assert getattr(a, f) == pytest.approx(getattr(b, f), abs=1e-12)


def test_skalieren_der_zeitachse_skaliert_nur_die_zeitpunkte():
    for s in datasets(count=30):
        a = M.curves(s, GRID)
        b = M.curves(Sample(s.time * 2.5, s.cause), GRID * 2.5)
        for f in ("cif_lkw", "cif_bahn", "remaining", "naive_lkw", "naive_bahn"):
            assert getattr(a, f) == pytest.approx(getattr(b, f), abs=1e-12)


def test_vertauschen_der_wege_vertauscht_die_kurven():
    for s in datasets(count=30):
        swapped = Sample(s.time, np.where(s.cause == LKW, BAHN, np.where(s.cause == BAHN, LKW, 0)))
        a, b = M.curves(s, GRID), M.curves(swapped, GRID)
        assert a.cif_lkw == pytest.approx(b.cif_bahn, abs=1e-12) and a.cif_bahn == pytest.approx(b.cif_lkw, abs=1e-12)
        assert a.naive_lkw == pytest.approx(b.naive_bahn, abs=1e-12) and a.remaining == pytest.approx(b.remaining, abs=1e-12)


def test_mehr_bahn_nimmt_dem_lkw_in_gepaarten_welten_nie_etwas_dazu():
    """Gepaarte Zufallszahlen: eine größere Bahn-Rate kann dem LKW keine Abholung hinzufügen, die kumulierte Inzidenz des LKW sinkt (oder bleibt) in jedem Datensatz."""
    from kkr_scenario import simulate

    for seed in range(20):
        a = M.estimate_fast(simulate(200, 0.10, 0.05, 15.0, seed), 5.0)
        b = M.estimate_fast(simulate(200, 0.10, 0.30, 15.0, seed), 5.0)
        assert b.cif_lkw <= a.cif_lkw + 1e-12
