"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen und Zusammenfassungen, nicht Seeds; der Test prüft damit die Grenzen und jede genannte Zahl direkt."""

from fractions import Fraction as F

import numpy as np
import pytest

import kkr_evaluation as ev
import kkr_model as M
from kkr_scenario import PATTERNS, World


def _summary(truth=0.30, aj=0.29, naive=0.40, naive_se=0.003, naive_above=190, reps=200, max_diff=0.1, below=0, aj_above=90):
    return M.Summary(reps=reps, truth=truth, truth_net=0.39, aj=M.Spread(aj, 0.03, 0.002, 0.24, 0.34), naive=M.Spread(naive, 0.04, naive_se, 0.33, 0.46),
                     naive_above_truth=naive_above, aj_above_truth=aj_above, naive_below_aj=below, max_naive_minus_aj=max_diff, censored=0.26)


# ------------------------------------------------------------------------------------------------ Formatierung


def test_formatierung_dezimalkomma_und_prozentpunkte():
    assert ev.num(0.58333, 3) == "0,583" and ev.pct(0.285398) == "28,5 %" and ev.pct(0.285398, 2) == "28,54 %"
    assert ev.pp(0.104) == "+10,4 Prozentpunkte" and ev.pp(-0.02) == "-2,0 Prozentpunkte" and ev.pts(-0.0021) == "0,21 Prozentpunkte"
    assert ev.frac(F(7, 12)) == "7/12" and ev.frac(F(0)) == "0" and ev.frac(F(2, 1)) == "2"


# ------------------------------------------------------------------------------------------------ Experiment 1


def test_verdikt_handbeispiel_warnt_und_nennt_brueche():
    kind, text = ev.verdict_hand(M.hand_table(PATTERNS["Sechs Container (Handbeispiel)"]))
    assert kind == "warning"
    assert "**7/12** ≈ 0,583" in text and "die der Bahn bei 5/12" in text
    assert "LKW 7/12 + Bahn 5/12 + noch da 0 = 1" in text
    assert "**1** ≈ 1,000" in text and "0,417 mehr" in text


def test_verdikt_handbeispiel_zensierung_vor_dem_ersten_ereignis():
    kind, text = ev.verdict_hand(M.hand_table(PATTERNS["Zensierung vor dem ersten Ereignis"]))
    assert kind == "warning" and "**1/2** ≈ 0,500" in text and "**3/5** ≈ 0,600" in text and "0,100 mehr" in text


def test_verdikt_handbeispiel_ohne_konkurrenz_ist_gleichheit():
    kind, text = ev.verdict_hand(M.hand_table(PATTERNS["Ohne Konkurrenz: die Bahn holt nie ab"]))
    assert kind == "success" and "**1** ≈ 1,000" in text and "1 − Kaplan–Meier ist 1: beide stimmen überein" in text


def test_verdikt_handbeispiel_bahn_nach_der_letzten_lkw_abholung_ist_ebenfalls_gleichheit():
    # LKW am Tag 2 und 4, die Bahn erst am Tag 6: vor der letzten LKW-Abholung holt die Bahn nichts ab
    kind, text = ev.verdict_hand(M.hand_table([(2, 1), (4, 1), (6, 2)]))
    assert kind == "success" and "beide stimmen überein" in text


def test_verdikt_handbeispiel_ohne_abholung():
    kind, text = ev.verdict_hand([])
    assert kind == "info" and "alle zensiert" in text


# ------------------------------------------------------------------------------------------------ Experiment 2


def test_verdikt_studie_warnt_ab_drei_standardfehlern_und_nennt_jede_zahl():
    w = World()
    kind, text = ev.verdict_study(_summary(truth=0.30, naive=0.40, naive_se=0.003), w)   # Abstand 0,10 = 33 Standardfehler
    assert kind == "warning"
    assert "Wahrheit 30,0 %" in text and "bei 29,0 %" in text and "bei 40,0 %" in text and "+10,0 Prozentpunkte über der Wahrheit" in text
    assert "In 190 von 200 Läufen" in text and "in 90 von 200" in text and "Standardfehler 0,30 Prozentpunkte" in text


def test_verdikt_studie_schwelle_drei_standardfehler():
    w = World()
    assert ev.verdict_study(_summary(truth=0.30, naive=0.3089, naive_se=0.003), w)[0] == "info"      # knapp unter 3 Standardfehlern
    assert ev.verdict_study(_summary(truth=0.30, naive=0.3091, naive_se=0.003), w)[0] == "warning"   # knapp darüber
    kind, text = ev.verdict_study(_summary(truth=0.30, naive=0.301, naive_se=0.003), w)
    assert kind == "info" and "+0,1 Prozentpunkte" in text


def test_verdikt_studie_ohne_bahn_meldet_gleichheit_nur_wenn_wirklich_gleich():
    w = World(h2=0.0)
    kind, text = ev.verdict_study(_summary(truth=0.39, aj=0.39, naive=0.39, max_diff=0.0, below=0), w)
    assert kind == "success" and "in allen 200 Läufen gleich" in text and "0.0e+00" in text
    # ein Unterschied im Lauf: kein Erfolgs-Verdikt
    assert ev.verdict_study(_summary(truth=0.39, aj=0.39, naive=0.39, max_diff=0.01), w)[0] != "success"
    assert ev.verdict_study(_summary(truth=0.39, aj=0.39, naive=0.39, max_diff=0.0, below=1), w)[0] != "success"


def test_verdikt_reihe_nennt_anfang_ende_und_spannen():
    rows = [{"h2": 0.0, "truth": 0.393, "aj": 0.392, "naive": 0.392, "reps": 200}, {"h2": 0.15, "truth": 0.285, "aj": 0.282, "naive": 0.389, "reps": 200},
            {"h2": 0.4, "truth": 0.184, "aj": 0.180, "naive": 0.386, "reps": 200}]
    kind, text = ev.verdict_h2_sweep(rows)
    assert kind == "info"
    assert "Von Bahn-Rate 0,00 bis 0,40 fällt die Wahrheit von 39,3 % auf 18,4 %" in text and "bei 39,2 % bis 18,0 %" in text
    assert "zwischen 38,6 % und 39,2 %" in text and "Mittel über 200 Läufe je Rate" in text


# ------------------------------------------------------------------------------------------------ Experiment 3


def _curves(naive_lkw, naive_bahn, cif_lkw, cif_bahn, remaining):
    grid = np.array([0.0, 5.0, 10.0])
    return M.Curves(grid=grid, cif_lkw=np.array(cif_lkw), cif_bahn=np.array(cif_bahn), remaining=np.array(remaining), naive_lkw=np.array(naive_lkw), naive_bahn=np.array(naive_bahn))


def test_verdikt_summe_warnt_bei_ueberschuss_und_nennt_die_summanden():
    c = _curves([0, 0.4, 0.6], [0, 0.5, 0.7], [0, 0.3, 0.4], [0, 0.4, 0.5], [1, 0.3, 0.1])
    kind, text = ev.verdict_sum(c, 5.0)
    assert kind == "warning"
    assert "Tag 5,0" in text and "LKW 40,0 % + Bahn 50,0 % + noch da 30,0 % = **120,0 %**" in text
    assert "30,0 % + 40,0 % + 30,0 % = 100,0 %" in text


def test_verdikt_summe_ohne_ueberschuss_ist_info():
    c = _curves([0, 0.2, 0.2], [0, 0.0, 0.0], [0, 0.2, 0.2], [0, 0.0, 0.0], [1, 0.8, 0.8])
    kind, text = ev.verdict_sum(c, 10.0)
    assert kind == "info" and "naive Summe ist 100,0 %" in text and "Tag 10,0" in text


# ------------------------------------------------------------------------------------------------ Experiment 4


def test_verdikt_zwei_fragen_nennt_ratenspanne_und_anteile():
    rows = [{"h2": 0.0, "truth": 0.393, "rate_lkw": 0.1000, "rate_true": 0.10}, {"h2": 0.15, "truth": 0.285, "rate_lkw": 0.0988, "rate_true": 0.10},
            {"h2": 0.4, "truth": 0.184, "rate_lkw": 0.0980, "rate_true": 0.10}]
    kind, text = ev.verdict_two_questions(rows)
    assert kind == "info"
    assert "Bahn-Raten von 0,00 bis 0,40" in text and "zwischen 0,098 und 0,100 je Tag" in text and "Wahrheit 0,10" in text
    assert "laut Wahrheit von 39,3 % auf 18,4 %" in text


def test_zeitraster_endet_nicht_hinter_der_letzten_beobachtung():
    from kkr_scenario import Sample

    s = Sample(np.array([1.0, 2.0, 3.0]), np.array([1, 0, 2]))
    g = ev.plot_grid(s, 5.0)
    assert g[0] == 0.0 and g[-1] == pytest.approx(3.0)
    s2 = Sample(np.array([1.0, 50.0]), np.array([1, 0]))
    assert ev.plot_grid(s2, 5.0)[-1] == pytest.approx(10.0)         # doppelter Zeitpunkt der Frage, mindestens 10 Tage
    assert ev.plot_grid(s2, 12.0)[-1] == pytest.approx(24.0)
