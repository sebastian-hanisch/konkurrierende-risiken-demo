"""Einheiten des Rechenkerns, jede an einer Mini-Instanz von Hand geprüft (DEMO-PLAYBOOK §1b): Handtabellen in Brüchen, geschlossene Wahrheit,
Schätzer, Datenerzeugung, Läufe."""

from fractions import Fraction as F

import numpy as np
import pytest

import kkr_core as K
import kkr_model as M
from kkr_scenario import BAHN, LKW, PATTERNS, Sample, World, make_sample, run_seed, simulate

# Von Hand gerechnet: (Tag, im Risiko, LKW, Bahn, S(t−), S(t), kumulierte Inzidenz LKW, Bahn, naiv 1 − Kaplan–Meier LKW)
HAND = {
    "Sechs Container (Handbeispiel)": [
        (2, 6, 1, 0, F(1), F(5, 6), F(1, 6), F(0), F(1, 6)),
        (3, 5, 0, 1, F(5, 6), F(2, 3), F(1, 6), F(1, 6), F(1, 6)),
        (4, 4, 1, 0, F(2, 3), F(1, 2), F(1, 3), F(1, 6), F(3, 8)),
        (6, 2, 0, 1, F(1, 2), F(1, 4), F(1, 3), F(5, 12), F(3, 8)),
        (8, 1, 1, 0, F(1, 4), F(0), F(7, 12), F(5, 12), F(1)),
    ],
    "Ohne Konkurrenz: die Bahn holt nie ab": [
        (2, 6, 1, 0, F(1), F(5, 6), F(1, 6), F(0), F(1, 6)),
        (4, 4, 1, 0, F(5, 6), F(5, 8), F(3, 8), F(0), F(3, 8)),
        (8, 1, 1, 0, F(5, 8), F(0), F(1), F(0), F(1)),
    ],
    "Gleichzeitig: LKW und Bahn am Tag 4": [
        (2, 6, 1, 0, F(1), F(5, 6), F(1, 6), F(0), F(1, 6)),
        (4, 5, 1, 1, F(5, 6), F(1, 2), F(1, 3), F(1, 6), F(1, 3)),
        (6, 2, 0, 1, F(1, 2), F(1, 4), F(1, 3), F(5, 12), F(1, 3)),
        (8, 1, 1, 0, F(1, 4), F(0), F(7, 12), F(5, 12), F(1)),
    ],
    "Zensierung vor dem ersten Ereignis": [
        (3, 5, 1, 0, F(1), F(4, 5), F(1, 5), F(0), F(1, 5)),
        (4, 4, 0, 1, F(4, 5), F(3, 5), F(1, 5), F(1, 5), F(1, 5)),
        (6, 2, 1, 0, F(3, 5), F(3, 10), F(1, 2), F(1, 5), F(3, 5)),
        (8, 1, 0, 1, F(3, 10), F(0), F(1, 2), F(1, 2), F(3, 5)),
    ],
}


@pytest.mark.parametrize("name", list(PATTERNS))
def test_handtabelle_jedes_musters_in_bruechen(name):
    table = M.hand_table(PATTERNS[name])
    want = HAND[name]
    assert len(table) == len(want)
    for row, w in zip(table, want):
        got = (row["t"], row["n"], row["d1"], row["d2"], row["s_prev"], row["s"], row["cif1"], row["cif2"], row["naive1"])
        assert got == w   # Brüche vergleichen exakt, auch gegen ganze Zahlen (F(2) == 2)


@pytest.mark.parametrize("name", list(PATTERNS))
def test_summe_der_drei_anteile_ist_in_jeder_zeile_exakt_eins(name):
    for row in M.hand_table(PATTERNS[name]):
        assert row["cif1"] + row["cif2"] + row["s"] == 1


def test_handbeispiel_der_planung_am_ende_7_12_gegen_naiv_1(six):
    last = M.hand_table(six)[-1]
    assert (last["cif1"], last["cif2"], last["naive1"]) == (F(7, 12), F(5, 12), F(1))


def test_handtabelle_gegen_den_kern_in_gleitkomma(six):
    t = [x for x, _ in six]
    c = [y for _, y in six]
    table = M.hand_table(six)
    grid = [float(r["t"]) for r in table]
    assert K.cumulative_incidence(t, c, 1, grid) == pytest.approx([float(r["cif1"]) for r in table], abs=1e-15)
    assert K.cumulative_incidence(t, c, 2, grid) == pytest.approx([float(r["cif2"]) for r in table], abs=1e-15)
    naive = M.naive_cif(Sample(np.array(t, float), np.array(c)), LKW, grid)
    assert naive == pytest.approx([float(r["naive1"]) for r in table], abs=1e-15)


def test_handtabelle_eingabepruefung():
    for rows in ([], [(0, 1)], [(2, 3)], [(-1, 1)]):
        with pytest.raises(K.SurvivalError):
            M.hand_table(rows)


def test_handtabelle_ohne_ereignis_ist_leer():
    assert M.hand_table([(2, 0), (3, 0)]) == []


# ------------------------------------------------------------------------------------------------ geschlossene Wahrheit


def test_wahrheit_handwerte():
    # h1 = 0,10, h2 = 0,15, t = 5: 0,4 · (1 − e^{−1,25}) = 0,4 · 0,7134952 = 0,2853981
    assert float(M.truth_cif(0.10, 0.15, 5)) == pytest.approx(0.4 * (1 - np.exp(-1.25)), abs=1e-15)
    assert float(M.truth_cif(0.10, 0.15, 5)) == pytest.approx(0.28540, abs=5e-6)
    assert float(M.truth_net(0.10, 5)) == pytest.approx(1 - np.exp(-0.5), abs=1e-15)
    assert float(M.truth_remaining(0.10, 0.15, 4)) == pytest.approx(np.exp(-1.0), abs=1e-15)


def test_wahrheit_summe_beider_wege_und_rest_ist_eins_und_ohne_raten_gibt_es_keine_abholung():
    for h1, h2 in ((0.1, 0.15), (0.3, 0.02), (0.02, 0.4), (0.1, 0.0)):
        t = np.array([0.0, 1.0, 5.0, 30.0])
        assert M.truth_cif(h1, h2, t) + M.truth_cif(h2, h1, t) + M.truth_remaining(h1, h2, t) == pytest.approx(np.ones(4), abs=1e-15)
    assert float(M.truth_cif(0.0, 0.0, 5.0)) == 0.0


def test_wahrheit_ohne_konkurrenz_ist_der_anteil_ohne_bahn():
    t = np.array([1.0, 5.0, 12.0])
    assert M.truth_cif(0.1, 0.0, t) == pytest.approx(M.truth_net(0.1, t), abs=1e-15)


# ------------------------------------------------------------------------------------------------ Schätzer


def test_schnelle_schaetzer_gleich_kern_auch_mit_gleichstaenden():
    rng = K.SplitMix64(1)
    for _ in range(150):
        n = int(5 + 40 * rng.uniform(1)[0])
        t = np.ceil(rng.uniform(n) * 8)             # ganze Tage: viele Gleichstände
        c = np.floor(rng.uniform(n) * 3).astype(int)
        s = Sample(t, c)
        t0 = float(1 + int(rng.uniform(1)[0] * 9))
        a, b = M.estimate_at(s, t0), M.estimate_fast(s, t0)
        for f in ("cif_lkw", "cif_bahn", "remaining", "naive_lkw", "naive_bahn"):
            assert getattr(a, f) == pytest.approx(getattr(b, f), abs=1e-12), f


def test_schaetzer_handwerte_der_sechs_container(six):
    s = Sample(np.array([x for x, _ in six], float), np.array([y for _, y in six]))
    for fn in (M.estimate_at, M.estimate_fast):
        e = fn(s, 4.0)
        assert (e.cif_lkw, e.cif_bahn, e.remaining, e.naive_lkw, e.naive_bahn) == pytest.approx((1 / 3, 1 / 6, 1 / 2, 3 / 8, 1 / 5), abs=1e-15)   # naive Bahn: Kaplan–Meier 4/5 nach dem Bahn-Ereignis am Tag 3 (n = 5)
        e = fn(s, 100.0)
        assert (e.cif_lkw, e.cif_bahn, e.remaining, e.naive_lkw) == pytest.approx((7 / 12, 5 / 12, 0.0, 1.0), abs=1e-15)


def test_schaetzer_vor_dem_ersten_ereignis_und_ohne_ereignis():
    s = Sample(np.array([2.0, 3.0, 4.0]), np.array([1, 2, 1]))
    for fn in (M.estimate_at, M.estimate_fast):
        e = fn(s, 1.0)
        assert (e.cif_lkw, e.cif_bahn, e.remaining, e.naive_lkw, e.naive_bahn) == (0.0, 0.0, 1.0, 0.0, 0.0)
    keins = Sample(np.array([2.0, 3.0]), np.array([0, 0]))
    for fn in (M.estimate_at, M.estimate_fast):
        e = fn(keins, 5.0)
        assert (e.cif_lkw, e.cif_bahn, e.remaining, e.naive_lkw, e.naive_bahn) == (0.0, 0.0, 1.0, 0.0, 0.0)


def test_nur_eine_ursache_beobachtet_gibt_der_anderen_den_anteil_null():
    s = Sample(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 1, 0, 1]))
    e = M.estimate_fast(s, 10.0)
    assert e.cif_bahn == 0.0 and e.naive_bahn == 0.0
    assert e.cif_lkw == pytest.approx(e.naive_lkw, abs=1e-15)   # ohne Konkurrenz: kumulierte Inzidenz = 1 − Kaplan–Meier


def test_schaetzer_eingabepruefung():
    for t, c in (([0.0, 1.0], [1, 1]), ([1.0, 2.0], [1, -1]), ([], [])):
        with pytest.raises(K.SurvivalError):
            M.estimate_fast(Sample(np.array(t, float), np.array(c, int)), 1.0)


def test_ursachenspezifische_rate_handwert(six):
    s = Sample(np.array([x for x, _ in six], float), np.array([y for _, y in six]))
    assert M.cause_rate(s, LKW) == pytest.approx(3 / 28, abs=1e-15)   # drei LKW-Abholungen, Summe der Zeiten 2+3+4+5+6+8 = 28
    assert M.cause_rate(s, BAHN) == pytest.approx(2 / 28, abs=1e-15)


def test_kurven_summe_ist_eins_und_naive_summe_liegt_nie_darunter():
    w = World()
    sample = make_sample(w)
    c = M.curves(sample, np.linspace(0.0, 12.0, 61))
    assert c.sum_correct == pytest.approx(np.ones(61), abs=1e-12)
    assert np.all(c.sum_naive >= c.sum_correct - 1e-12)
    assert np.all(c.naive_lkw >= c.cif_lkw - 1e-12) and np.all(c.naive_bahn >= c.cif_bahn - 1e-12)


# ------------------------------------------------------------------------------------------------ Datenerzeugung


def test_simulation_ist_deterministisch_und_gueltig():
    a, b = simulate(300, 0.1, 0.15, 15.0, 7), simulate(300, 0.1, 0.15, 15.0, 7)
    assert np.array_equal(a.time, b.time) and np.array_equal(a.cause, b.cause)
    assert not np.array_equal(a.time, simulate(300, 0.1, 0.15, 15.0, 8).time)
    assert a.time.shape == (300,) and np.all(a.time > 0) and np.all(a.time <= 15.0) and set(a.cause.tolist()) <= {0, 1, 2}
    K.check_data(a.time, a.cause > 0)   # die Eingabeprüfung des Kerns akzeptiert den Datensatz


def test_simulation_ohne_bahn_hat_keine_bahn_abholung_und_ohne_raten_nur_zensierte():
    s = simulate(500, 0.1, 0.0, 15.0, 3)
    assert not np.any(s.cause == BAHN) and np.any(s.cause == LKW)
    nichts = simulate(200, 0.0, 0.0, 15.0, 3)
    assert np.all(nichts.cause == 0) and np.all(np.isfinite(nichts.time))


def test_gepaarte_zufallszahlen_mehr_bahn_nimmt_dem_lkw_nur_ab():
    wenig, viel = simulate(500, 0.1, 0.05, 15.0, 9), simulate(500, 0.1, 0.40, 15.0, 9)
    assert set(np.flatnonzero(viel.cause == LKW)) <= set(np.flatnonzero(wenig.cause == LKW))
    zensiert = (wenig.cause == 0) & (viel.cause == 0)
    assert np.array_equal(wenig.time[zensiert], viel.time[zensiert])   # dieselbe Zensierzeit


def test_simulation_eingabepruefung():
    for args in ((0, 0.1, 0.1, 10.0), (10, -0.1, 0.1, 10.0), (10, 0.1, -0.1, 10.0), (10, 0.1, 0.1, 0.0)):
        with pytest.raises(K.SurvivalError):
            simulate(*args, seed=1)


def test_run_seed_gibt_getrennte_ströme():
    seeds = {run_seed(35, r) for r in range(200)}
    assert len(seeds) == 200 and run_seed(35, 0) == 35


# ------------------------------------------------------------------------------------------------ Läufe und Reihen


def test_studie_ist_deterministisch_und_hat_je_lauf_einen_wert():
    a, b = M.run_study(World(n=100), 20), M.run_study(World(n=100), 20)
    assert a.reps == 20 and np.array_equal(a.cif_lkw, b.cif_lkw) and np.array_equal(a.naive_lkw, b.naive_lkw)
    assert a.cif_lkw.shape == a.naive_lkw.shape == a.rate_lkw.shape == a.censored.shape == (20,)


def test_studie_eingabepruefung():
    with pytest.raises(K.SurvivalError):
        M.run_study(World(), 0)


def test_zusammenfassung_zaehlt_gegen_die_wahrheit():
    study = M.run_study(World(), 60)
    s = M.summarize(study)
    assert s.truth == pytest.approx(0.285398, abs=1e-6) and s.truth_net == pytest.approx(0.393469, abs=1e-6)
    assert s.naive_above_truth == int(np.sum(study.naive_lkw > s.truth)) and s.aj_above_truth == int(np.sum(study.cif_lkw > s.truth))
    assert s.naive_below_aj == 0 and s.max_naive_minus_aj >= 0.0


def test_spread_handwerte():
    sp = M.spread([1.0, 2.0, 3.0, 4.0])
    assert sp.mean == 2.5 and sp.sd == pytest.approx(np.sqrt(5 / 3), abs=1e-15) and sp.se == pytest.approx(np.sqrt(5 / 3) / 2, abs=1e-15)
    assert M.spread([2.0]).sd == 0.0


def test_reihe_ueber_die_bahn_rate_hat_je_wert_eine_zeile_und_gleiche_welt_sonst():
    rows = M.sweep_h2(World(n=100), (0.0, 0.1, 0.4), 20)
    assert [r["h2"] for r in rows] == [0.0, 0.1, 0.4]
    assert rows[0]["naive"] == pytest.approx(rows[0]["aj"], abs=1e-12)   # ohne Bahn: beide Schätzer gleich
    assert all(r["reps"] == 20 for r in rows)
    assert rows[0]["truth"] > rows[1]["truth"] > rows[2]["truth"]
