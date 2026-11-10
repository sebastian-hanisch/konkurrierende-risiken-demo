"""Abbildungen: jede Figur lässt sich für die Standardwelt und die Ränder bauen, alle Achsen sind gesperrt, die Zahlen im Diagramm sind die der Modelle."""

import numpy as np
import pytest

import kkr_evaluation as ev
import kkr_model as M
import kkr_visualization as V
from kkr_scenario import PATTERNS, World


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


@pytest.mark.parametrize("world", [World(), World(h2=0.0), World(h1=0.02, h2=0.4, n=50, window=5.0, t0=30.0), World(n=600, window=1000.0, t0=1.0)])
def test_alle_figuren_der_experimente_lassen_sich_bauen_und_sind_gesperrt(world):
    study = M.run_study(world, 10)
    summary = M.summarize(study)
    sample, curves = ev.analyse_curves(world)
    sweep = M.sweep_h2(world, (0.0, 0.2), 5)
    for fig in (V.build_estimators(study, summary), V.build_h2_sweep(sweep), V.build_curves(curves, world), V.build_sum(curves, world), V.build_two_questions(sweep, world.t0)):
        assert _locked(fig) and len(fig.data) >= 2


@pytest.mark.parametrize("name", list(PATTERNS))
def test_handdiagramm_zeigt_die_werte_der_handtabelle(name):
    table = M.hand_table(PATTERNS[name])
    fig = V.build_hand(table, float(table[-1]["t"]))
    assert _locked(fig)
    cif1 = next(t for t in fig.data if t.name.startswith("LKW"))
    assert cif1.y[-1] == pytest.approx(float(table[-1]["cif1"]), abs=1e-15) and cif1.y[0] == 0.0
    naive = next(t for t in fig.data if t.name.startswith("naiv"))
    assert naive.y[-1] == pytest.approx(float(table[-1]["naive1"]), abs=1e-15)


def test_handdiagramm_ohne_ereignis_ist_leer_aber_kein_fehler():
    assert V.build_hand([], None).data == ()


def test_kurven_im_diagramm_sind_die_der_schaetzer():
    world = World()
    _, curves = ev.analyse_curves(world)
    fig = V.build_curves(curves, world)
    naive = next(t for t in fig.data if t.name.startswith("LKW naiv"))
    assert np.array_equal(np.asarray(naive.y), curves.naive_lkw)
