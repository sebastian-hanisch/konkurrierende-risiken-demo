"""Orakel-Tests der kumulierten Inzidenz und der naiven Rechnung: jeweils ein ANDERER Rechenweg als der Kern und die Demo (DEMO-PLAYBOOK §12).

Drei unabhängige Wege:
  1. Umverteilung nach rechts (Efron 1967, nur als Rechenweg gebraucht, nicht aus der Quelle zitiert): Jeder Container trägt zunächst die Masse 1/n; die Masse eines
     zensierten Containers wird zu gleichen Teilen auf alle Container verteilt, die danach kommen. Die Masse der Abholungen je Weg ergibt die kumulierte Inzidenz.
     Keine Risikomengen, kein Produkt, keine Summe über Ereigniszeiten: ein anderer Algorithmus, in exakten Brüchen, an Daten mit Gleichständen.
  2. Geschlossene Formel bei konstanten Raten, per Quadratur des Integrals (scipy) und per große Simulation mit Band (n = 40 000, feste Zufalls-Seeds).
  3. Die Handtabelle der Demo (eigene Bruchrechnung) gegen den Weg 1, Bruch für Bruch.
Randfälle: Gleichstände, Zensierung vor dem ersten Ereignis, nur eine Ursache, alle zensiert."""

from fractions import Fraction as F

import numpy as np
import pytest

import kkr_core as K
import kkr_model as M
from kkr_scenario import BAHN, LKW, PATTERNS, Sample, World, simulate


def redistribution(time, cause, k, grid):
    """Kumulierte Inzidenz des Wegs k durch Umverteilung nach rechts, in exakten Brüchen. Gleichstand: Abholungen vor Zensierungen (Konvention des Kerns).
    Gibt außerdem den Anteil zurück, der zu jedem Rasterpunkt noch nicht abgeholt ist (1 − Masse aller Abholungen bis dahin)."""
    n = len(time)
    order = sorted(range(n), key=lambda i: (time[i], 0 if cause[i] > 0 else 1))
    mass = [F(1, n)] * n
    for pos, i in enumerate(order):
        if cause[i] == 0:
            later = order[pos + 1 :]
            if later:
                share = mass[i] / len(later)
                for j in later:
                    mass[j] += share
            mass[i] = F(0)   # die Masse ist weitergegeben (oder, hinter dem letzten Container, verloren: sie bleibt „noch da“)
    cif, rest = [], []
    for g in grid:
        cif.append(sum((mass[i] for i in range(n) if cause[i] == k and time[i] <= g), F(0)))
        rest.append(1 - sum((mass[i] for i in range(n) if cause[i] > 0 and time[i] <= g), F(0)))
    return cif, rest


def random_small(rng, n_max=9):
    n = 3 + int(rng.uniform(1)[0] * (n_max - 2))
    t = np.ceil(rng.uniform(n) * 7).astype(int)                # ganze Tage 1 bis 7: viele Gleichstände
    c = np.floor(rng.uniform(n) * 3).astype(int)               # 0 zensiert, 1 LKW, 2 Bahn
    return t.tolist(), c.tolist()


# ------------------------------------------------------------------------------------------------ Weg 1: Umverteilung nach rechts


def test_umverteilung_am_handbeispiel(six):
    t, c = [x for x, _ in six], [y for _, y in six]
    cif1, rest = redistribution(t, c, 1, [2, 3, 4, 6, 8])
    assert cif1 == [F(1, 6), F(1, 6), F(1, 3), F(1, 3), F(7, 12)]
    assert redistribution(t, c, 2, [8])[0] == [F(5, 12)]
    assert rest == [F(5, 6), F(2, 3), F(1, 2), F(1, 4), F(0)]
    # naiv: die Bahn als zensiert abgebildet
    naive, _ = redistribution(t, [1 if y == 1 else 0 for y in c], 1, [2, 4, 8])
    assert naive == [F(1, 6), F(3, 8), F(1)]


def test_kumulierte_inzidenz_und_schnelle_fassung_gegen_umverteilung_in_200_zufallsdaten():
    rng = K.SplitMix64(2026)
    for _ in range(200):
        t, c = random_small(rng)
        grid = sorted(set(t)) + [max(t) + 1]
        for k in (1, 2):
            want, _ = redistribution(t, c, k, grid)
            got = K.cumulative_incidence(t, c, k, grid)
            assert got == pytest.approx([float(x) for x in want], abs=1e-12), (t, c, k)


def test_naive_rechnung_gegen_umverteilung_in_200_zufallsdaten():
    rng = K.SplitMix64(2027)
    for _ in range(200):
        t, c = random_small(rng)
        s = Sample(np.array(t, float), np.array(c))
        for k in (1, 2):
            grid = sorted(set(t))
            want, _ = redistribution(t, [x if x == k else 0 for x in c], k, grid)
            assert M.naive_cif(s, k, grid) == pytest.approx([float(x) for x in want], abs=1e-12), (t, c, k)


def test_schnelle_schaetzer_und_noch_da_gegen_umverteilung_in_200_zufallsdaten():
    rng = K.SplitMix64(2028)
    for _ in range(200):
        t, c = random_small(rng)
        s = Sample(np.array(t, float), np.array(c))
        for t0 in sorted(set(t)) + [0.5]:
            e = M.estimate_fast(s, float(t0))
            cif1, rest = redistribution(t, c, 1, [t0])
            cif2, _ = redistribution(t, c, 2, [t0])
            assert (e.cif_lkw, e.cif_bahn) == pytest.approx((float(cif1[0]), float(cif2[0])), abs=1e-12), (t, c, t0)
            assert e.remaining == pytest.approx(float(rest[0]), abs=1e-12), (t, c, t0)


def test_handtabelle_der_demo_gegen_umverteilung_bruch_fuer_bruch():
    for name, rows in PATTERNS.items():
        t, c = [x for x, _ in rows], [y for _, y in rows]
        table = M.hand_table(rows)
        grid = [r["t"] for r in table]
        cif1, rest = redistribution(t, c, 1, grid)
        cif2, _ = redistribution(t, c, 2, grid)
        naive, _ = redistribution(t, [1 if y == 1 else 0 for y in c], 1, grid)
        assert [r["cif1"] for r in table] == cif1, name
        assert [r["cif2"] for r in table] == cif2, name
        assert [r["s"] for r in table] == rest, name
        assert [r["naive1"] for r in table] == naive, name


def test_randfaelle_des_orakels_nur_eine_ursache_alle_zensiert_zensierung_vor_dem_ersten_ereignis():
    # nur eine Ursache: die Bahn hat den Anteil 0, die kumulierte Inzidenz des LKW ist 1 − Kaplan–Meier
    cif1, _ = redistribution([1, 2, 3, 4], [1, 1, 0, 1], 1, [4])
    cif2, _ = redistribution([1, 2, 3, 4], [1, 1, 0, 1], 2, [4])
    assert cif2 == [F(0)] and cif1 == [F(1)]   # Kaplan–Meier: 3/4, 1/2 (nach Tag 2), 0 (nach Tag 4, nur noch ein Container im Risiko), also Anteil 1
    # alle zensiert: nichts wird abgeholt
    assert redistribution([1, 2], [0, 0], 1, [5]) == ([F(0)], [F(1)])
    # Zensierung vor dem ersten Ereignis: 1 wird zensiert, die übrigen vier teilen die Masse 1/5 gleich auf
    cif1, rest = redistribution([1, 3, 4, 5, 6], [0, 1, 2, 0, 1], 1, [3])
    assert cif1 == [F(1, 5) + F(1, 20)] and rest == [1 - F(1, 5) - F(1, 20)]


# ------------------------------------------------------------------------------------------------ Weg 2: geschlossene Formel


def test_geschlossene_wahrheit_gegen_quadratur():
    quad = pytest.importorskip("scipy.integrate").quad
    for h1, h2 in ((0.10, 0.15), (0.02, 0.40), (0.30, 0.0), (0.05, 0.05)):
        for t in (0.5, 5.0, 12.0, 30.0):
            want, _ = quad(lambda u: h1 * np.exp(-(h1 + h2) * u), 0.0, t)
            assert float(M.truth_cif(h1, h2, t)) == pytest.approx(want, abs=1e-10)


def test_simulation_trifft_die_formel_und_die_naive_rechnung_den_anteil_ohne_bahn():
    # n = 40 000 Container, ein Datensatz: Standardfehler der kumulierten Inzidenz etwa 0,3 Prozentpunkte (gemessen: Streuung bei n = 300 rund 2,9 Punkte, skaliert mit 1/√n)
    w = World(n=40000, seed=11)
    e = M.estimate_fast(simulate(w.n, w.h1, w.h2, w.window, w.seed), w.t0)
    assert e.cif_lkw == pytest.approx(float(M.truth_cif(w.h1, w.h2, w.t0)), abs=0.012)
    assert e.naive_lkw == pytest.approx(float(M.truth_net(w.h1, w.t0)), abs=0.015)
    assert e.naive_lkw - e.cif_lkw > 0.08              # die naive Rechnung liegt klar über der Wahrheit
    assert e.cif_lkw + e.cif_bahn + e.remaining == pytest.approx(1.0, abs=1e-12)
    assert e.remaining == pytest.approx(float(M.truth_remaining(w.h1, w.h2, w.t0)), abs=0.012)


def test_ursachenspezifische_rate_gegen_die_wahre_rate_unabhaengig_von_der_bahn():
    for h2 in (0.0, 0.15, 0.40):
        s = simulate(40000, 0.10, h2, 15.0, 5)
        assert M.cause_rate(s, LKW) == pytest.approx(0.10, abs=0.006)
        if h2 > 0:
            assert M.cause_rate(s, BAHN) == pytest.approx(h2, abs=0.01)


def test_summe_der_wahrheit_wege_gegen_simulation_ohne_zensierung():
    # ohne Zensierung (Fenster 1000 Tage) ist der Anteil per LKW bis Tag 60 fast h1/(h1+h2): jeder Container ist bis dahin abgeholt
    s = simulate(20000, 0.10, 0.15, 1000.0, 2)
    assert float(np.mean(s.cause == LKW)) == pytest.approx(0.10 / 0.25, abs=0.012)
    assert float(M.truth_cif(0.10, 0.15, 60.0)) == pytest.approx(0.4, abs=2e-6)
