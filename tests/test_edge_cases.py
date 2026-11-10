"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstände, nur eine Ursache, Zensierung vor dem ersten Ereignis,
alle zensiert, Grenzen der Regler, Meldungen statt stiller Zahlen."""

import numpy as np
import pytest

import kkr_constants as C
import kkr_core as K
import kkr_evaluation as ev
import kkr_model as M
from kkr_scenario import BAHN, LKW, Sample, World, simulate


def _s(times, causes):
    return Sample(np.array(times, float), np.array(causes, int))


def test_ein_container_und_genau_zwei_container():
    for fn in (M.estimate_at, M.estimate_fast):
        e = fn(_s([3.0], [LKW]), 5.0)
        assert (e.cif_lkw, e.cif_bahn, e.remaining, e.naive_lkw) == (1.0, 0.0, 0.0, 1.0)
        e = fn(_s([3.0, 4.0], [LKW, BAHN]), 5.0)
        assert (e.cif_lkw, e.cif_bahn, e.remaining) == pytest.approx((0.5, 0.5, 0.0), abs=1e-15)
        assert e.naive_lkw == pytest.approx(0.5, abs=1e-15) and e.naive_bahn == 1.0   # naiv: der einzige Bahn-Container ist am Tag 4 der letzte im Risiko, also 1
    assert M.hand_table([(3, LKW)])[0]["cif1"] == 1


def test_gleichzeitige_abholungen_beider_wege_zaehlen_beide():
    # zwei Container am Tag 2, einer per LKW und einer per Bahn, der dritte am Tag 5 per LKW: Tag 2: je 1/3, Tag 5: 1/3 + 1/3 für den LKW
    for fn in (M.estimate_at, M.estimate_fast):
        e = fn(_s([2.0, 2.0, 5.0], [LKW, BAHN, LKW]), 2.0)
        assert (e.cif_lkw, e.cif_bahn, e.remaining) == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=1e-15)
        e = fn(_s([2.0, 2.0, 5.0], [LKW, BAHN, LKW]), 5.0)
        assert (e.cif_lkw, e.cif_bahn, e.remaining) == pytest.approx((2 / 3, 1 / 3, 0.0), abs=1e-15)


def test_ereignis_und_zensierung_am_selben_tag_ereignis_zuerst():
    # Tag 3: ein LKW-Container wird abgeholt, einer ist zensiert; der zensierte bleibt am Tag 3 im Risiko (n = 3, nicht 2)
    e = M.estimate_fast(_s([3.0, 3.0, 6.0], [LKW, 0, BAHN]), 3.0)
    assert e.cif_lkw == pytest.approx(1 / 3, abs=1e-15)


def test_zensierung_vor_dem_ersten_ereignis_aendert_nur_die_risikomenge():
    e = M.estimate_fast(_s([1.0, 3.0, 4.0, 5.0, 6.0], [0, LKW, BAHN, 0, LKW]), 3.0)
    assert e.cif_lkw == pytest.approx(1 / 4, abs=1e-15)    # n = 4 am Tag 3, nicht 5
    assert M.estimate_fast(_s([1.0, 2.0], [0, 0]), 5.0).cif_lkw == 0.0


def test_alle_zensiert_gibt_nullen_und_keine_fehler():
    s = _s([1.0, 2.0, 3.0], [0, 0, 0])
    c = M.curves(s, np.linspace(0, 5, 11))
    assert np.all(c.cif_lkw == 0) and np.all(c.cif_bahn == 0) and np.all(c.remaining == 1) and np.all(c.naive_lkw == 0)
    assert ev.verdict_sum(c, 3.0)[0] == "info"


def test_nur_eine_ursache_beobachtet_kumulierte_inzidenz_gleich_eins_minus_kaplan_meier():
    s = _s([1.0, 2.0, 2.0, 3.0, 7.0, 9.0], [LKW, LKW, 0, LKW, 0, LKW])
    c = M.curves(s, np.linspace(0, 10, 41))
    assert np.all(c.cif_bahn == 0) and c.cif_lkw == pytest.approx(c.naive_lkw, abs=1e-15)


def test_zeitpunkt_der_frage_hinter_dem_beobachtungsfenster_fuehrt_zu_keinem_fehler():
    # Fenster 5 Tage, Frage an Tag 30: nach der letzten Abholung bleibt die Kurve konstant (kein Fehler, keine NaN)
    w = World(window=5.0, t0=30.0)
    study = M.run_study(w, 20)
    assert np.all(np.isfinite(study.cif_lkw)) and np.all(np.isfinite(study.naive_lkw))


def test_regler_grenzen_laufen_ohne_fehler_und_geben_endliche_werte():
    for h1, h2, n, window, t0 in ((C.H1_MIN, C.H2_MIN, C.N_MIN, 5, C.T0_MIN), (C.H1_MAX, C.H2_MAX, C.N_MAX, 1000, C.T0_MAX),
                                  (C.H1_MIN, C.H2_MAX, C.N_MIN, 5, C.T0_MAX), (C.H1_MAX, C.H2_MIN, C.N_MAX, 1000, C.T0_MIN)):
        w = World(h1=h1, h2=h2, n=n, window=float(window), t0=float(t0))
        s = M.summarize(M.run_study(w, 10))
        assert all(np.isfinite(x) for x in (s.truth, s.aj.mean, s.naive.mean, s.censored))
        for kind, text in (ev.verdict_study(s, w),):
            assert kind in ("success", "info", "warning") and text
        assert ev.verdict_h2_sweep(M.sweep_h2(w, (0.0, 0.4), 10))[1]


def test_fehler_des_kerns_sind_survival_error_mit_verstaendlicher_meldung():
    with pytest.raises(K.SurvivalError, match="größer als 0"):
        M.estimate_fast(_s([0.0, 1.0], [1, 1]), 1.0)
    with pytest.raises(K.SurvivalError, match="Container"):
        simulate(0, 0.1, 0.1, 10.0, 1)
    with pytest.raises(K.SurvivalError, match="Ursache"):
        M.hand_table([(1, 5)])
