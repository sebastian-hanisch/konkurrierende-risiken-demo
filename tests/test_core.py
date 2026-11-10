"""Tests der kopierten Kernfunktionen (aus ueberlebenszeit-planung/kern/test_uz_core.py übernommen, soweit die Demo die Funktionen braucht)."""

from fractions import Fraction as F

import numpy as np
import pytest

import kkr_core as K

SIX_T = [2, 3, 4, 5, 5, 8]
SIX_E = [1, 0, 1, 1, 0, 1]


def test_km_handbeispiel_sechs_container():
    km = K.kaplan_meier(SIX_T, SIX_E)
    assert km["t"].tolist() == [2, 4, 5, 8]
    assert km["n"].tolist() == [6, 4, 3, 1]
    for got, want in zip(km["s"], [F(5, 6), F(5, 8), F(5, 12), F(0)]):
        assert got == pytest.approx(float(want), abs=1e-15)


def test_km_ohne_zensierung_ist_eins_minus_verteilungsfunktion():
    rng = K.SplitMix64(1)
    t = np.round(rng.exponential(50) * 3 + 0.5, 1)  # mit Gleichständen
    km = K.kaplan_meier(t, np.ones(50, bool))
    for x in (0.5, 1.0, 2.0, 4.0, 8.0):
        assert K.survival_at(km, x) == pytest.approx(float(np.mean(t > x)), abs=1e-12)


def test_km_alle_zensiert():
    km = K.kaplan_meier([1, 2, 3], [0, 0, 0])
    assert km["t"].size == 0 and K.survival_at(km, 5.0) == 1.0


def test_eingabepruefung():
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([3, 3, 5], [1, 1, 1], [3, 3, 0])  # Eintritt = Zeit
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([0, 1], [1, 1])  # Zeit 0
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([1, 2], [1, 2])  # Ereignis 2
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([1, 2], [1])  # Länge
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([], [])


def test_konkurrierende_risiken_handbeispiel():
    t = [2, 3, 4, 5, 6, 8]
    c = [1, 2, 1, 0, 2, 1]
    lkw = K.cumulative_incidence(t, c, 1, [2, 3, 4, 6, 8, 10])
    bahn = K.cumulative_incidence(t, c, 2, [2, 3, 4, 6, 8, 10])
    assert lkw.tolist() == pytest.approx([1 / 6, 1 / 6, 1 / 3, 1 / 3, 7 / 12, 7 / 12], abs=1e-15)
    assert bahn[-1] == pytest.approx(5 / 12, abs=1e-15)
    assert lkw[-1] + bahn[-1] == pytest.approx(1.0, abs=1e-15)
    naiv = K.kaplan_meier(t, [1 if x == 1 else 0 for x in c])
    assert 1 - K.survival_at(naiv, 8.0) == 1.0  # naiv: 1 statt 7/12


def test_cumulative_incidence_eingabepruefung():
    for t, c in (([], []), ([1, 2], [1]), ([0, 1], [1, 1]), ([1, 2], [1, -1]), ([1, float("nan")], [1, 1])):
        with pytest.raises(K.SurvivalError):
            K.cumulative_incidence(t, c, 1, [1.0])


def test_cif_geschlossene_formel_bei_konstanten_raten():
    import math

    rng = K.SplitMix64(7)
    n = 20000
    h1, h2 = 0.10, 0.15
    t1, t2 = rng.exponential(n) / h1, rng.exponential(n) / h2
    c = rng.uniform(n) * 15 + 1e-9
    t = np.minimum(np.minimum(t1, t2), c)
    cause = np.where(np.minimum(t1, t2) <= c, np.where(t1 < t2, 1, 2), 0)
    got = K.cumulative_incidence(t, cause, 1, [5.0])[0]
    want = h1 / (h1 + h2) * (1 - math.exp(-(h1 + h2) * 5))
    assert got == pytest.approx(want, abs=0.01)


def test_splitmix_referenzwert_und_determinismus():
    # erster Ausgabewert von SplitMix64 bei Seed 0 (Referenz: Vigna, splitmix64.c)
    assert int(K.SplitMix64(0).u64(1)[0]) == 0xE220A8397B1DCDAF
    a, b = K.SplitMix64(5), K.SplitMix64(5)
    assert np.array_equal(a.uniform(10), b.uniform(10))
    u = K.SplitMix64(9).uniform(50000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01
    x = K.SplitMix64(9).exponential(50000)
    assert abs(x.mean() - 1.0) < 0.03


def test_kopie_stimmt_mit_der_quelle_ueberein_falls_vorhanden():
    """Die Funktionen dieser Datei sind wörtlich aus ueberlebenszeit-planung/kern/uz_core.py kopiert; gibt es die Quelle, stimmen die Texte der Funktionen überein."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "ueberlebenszeit-planung" / "kern" / "uz_core.py"
    if not src.exists():
        pytest.skip("ueberlebenszeit-planung/kern/uz_core.py nicht vorhanden")
    mine = Path(K.__file__).read_text(encoding="utf-8")
    theirs = src.read_text(encoding="utf-8")

    def definitions(text):
        return {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    a, b = definitions(mine), definitions(theirs)
    assert set(a) <= set(b)
    assert {name for name in a if a[name] != b[name]} == set()
