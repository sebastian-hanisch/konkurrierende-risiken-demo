"""Jede Zahl der README steht in einer Messung (`tools/measure_konkurrierende_risiken_result.json`, eingefroren, feste Zufalls-Seeds) oder in einem Test.

Dieser Test prüft viererlei:
  1. Alle Messzellen werden mit dem Code der Demo neu gerechnet und stimmen mit der eingefrorenen Datei überein (Band 1e-9, Zählungen exakt).
  2. Die Tabellen der README stehen wörtlich so in der README, wie `tools/measure_konkurrierende_risiken.tables()` sie aus der Datei bildet.
  3. Die Prosa-Zahlen der README (Mittel, Zählungen, Spannen) folgen aus denselben Zahlen, jeweils mit ihren Voraussetzungen.
  4. Die Aussagen der Preset-Hilfen und des Annahmen-Textes gelten gegen die Messzellen; der Annahmen-Text steht gleich in App und README."""

import json
from pathlib import Path

import pytest

import kkr_constants as C
import kkr_evaluation as ev
import measure_konkurrierende_risiken as MC

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
DATA = json.loads((ROOT / "tools" / "measure_konkurrierende_risiken_result.json").read_text(encoding="utf-8"))
S = DATA["standard"]


def _same(a, b, path=""):
    """Rekursiver Vergleich: Zahlen mit Band, Ganzzahlen und Texte exakt."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            _same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert b == pytest.approx(a, rel=1e-9, abs=1e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


@pytest.fixture(scope="module")
def fresh():
    return MC.measure()


# ------------------------------------------------------------------------------------------------ 1. neu gerechnet


def test_alle_messzellen_neu_gerechnet_stimmen_mit_der_eingefrorenen_datei(fresh):
    _same(DATA, fresh)


# ------------------------------------------------------------------------------------------------ 2. Tabellen und Texte


def test_alle_tabellen_der_readme_stehen_woertlich_in_der_readme():
    for name, table in MC.tables(DATA).items():
        assert table in README, name


def test_annahmen_text_steht_gleich_in_app_und_readme():
    assert C.ASSUMPTIONS_MARKDOWN.rstrip() in README
    assert "C.ASSUMPTIONS_MARKDOWN" in (ROOT / "app.py").read_text(encoding="utf-8")


def test_readme_nennt_die_annahme_der_unabhaengigen_zensierung_und_die_linie():
    assert "Die Zensierung ist unabhängig vom Ereignis" in README and "**Überlebenszeit**" in README


# ------------------------------------------------------------------------------------------------ 3. Prosa-Zahlen


def test_standardwelt_zahlen_der_readme():
    assert (S["h1"], S["h2"], S["n"], S["window"], S["t0"], S["seed"], S["reps"]) == (0.10, 0.15, 300, 15, 5, 35, 200)
    assert ev.pct(S["truth"]) == "28,5 %" and ev.pct(S["truth_net"]) == "39,3 %" and ev.pct(S["censored"]) == "26,0 %"
    assert ev.pct(S["aj_mean"]) == "28,2 %" and ev.pct(S["naive_mean"]) == "38,9 %"
    assert ev.pts(S["aj_se"]) == "0,21 Prozentpunkte" and ev.pts(S["naive_se"]) == "0,29 Prozentpunkte"
    assert ev.pts(S["aj_sd"]) == "2,93 Prozentpunkte" and ev.pts(S["naive_sd"]) == "4,15 Prozentpunkte"
    assert ev.pp(S["naive_mean"] - S["truth"]) == "+10,4 Prozentpunkte" and ev.pct((S["naive_mean"] - S["truth"]) / S["truth"], 0) == "36 %"
    assert (S["naive_above_truth"], S["aj_above_truth"], S["naive_below_aj"]) == (199, 88, 0)
    assert ev.pts(S["max_naive_minus_aj"], 1) == "15,1 Prozentpunkte"
    assert ev.pts(S["aj_mean"] - S["truth"], 1) == "0,3 Prozentpunkte"
    assert 1.4 < (S["truth"] - S["aj_mean"]) / S["aj_se"] < 1.6          # „rund 1,5 Standardfehler“
    for needle in ("28,5 %", "38,9 %", "28,2 %", "26,0 %", "199", "88", "10,4 Prozentpunkte", "36 %", "15,1 Prozentpunkte", "rund 1,5 Standardfehler"):
        assert needle in README, needle


def test_plan_vergleich_steht_in_der_readme_mit_der_quelle_der_zahlen():
    assert "Vormessung: naive Rechnung 0,40 (+40 %), kumulierte Inzidenz 0,29, Wahrheit 0,285" in README and "out_uz_v1_bis_v8.txt" in README
    assert S["naive_mean"] < 0.395 and S["naive_mean"] != pytest.approx(0.40, abs=0.005)   # die Demo-Zahl weicht von 0,40 ab: es gilt die eigene


def test_stabilitaet_ueber_sechs_zufalls_seeds():
    st = DATA["stability"]
    assert [r["seed"] for r in st] == [35, 101, 202, 303, 404, 505] and all(r["reps"] == 200 for r in st)
    assert ev.pct(min(r["naive_mean"] for r in st)) == "38,9 %" and ev.pct(max(r["naive_mean"] for r in st)) == "39,5 %"
    assert ev.pct(min(r["aj_mean"] for r in st)) == "28,2 %" and ev.pct(max(r["aj_mean"] for r in st)) == "28,6 %"
    assert [r["naive_above_truth"] for r in st] == [199, 199, 200, 200, 200, 200] and min(r["aj_above_truth"] for r in st) == 88 and max(r["aj_above_truth"] for r in st) == 102
    assert "199 bis 200 von je 200 Läufen" in README and "88 bis 102" in README


def test_reihe_ueber_die_bahn_rate():
    rows = DATA["sweep_h2"]
    assert [r["h2"] for r in rows] == list(C.SWEEP_H2) and all(r["reps"] == 200 for r in rows)
    assert ev.pct(min(r["naive"] for r in rows)) == "38,6 %" and ev.pct(max(r["naive"] for r in rows)) == "39,3 %"
    assert ev.pct(rows[0]["truth"]) == "39,3 %" and ev.pct(rows[-1]["truth"]) == "18,4 %"
    # ohne Bahn: beide Schätzer gleich, in allen Läufen (Preset-Zelle) und im Mittel (Reihe)
    assert rows[0]["aj"] == pytest.approx(rows[0]["naive"], abs=1e-12)
    ohne = DATA["presets"]["Ohne Konkurrenz (Bahn-Rate 0)"]
    assert ohne["max_naive_minus_aj"] < 1e-12 and ohne["naive_below_aj"] == 0 and ohne["h2"] == 0.0
    assert rows[0]["naive_above_truth"] == ohne["naive_above_truth"] == 103 and 100 - 7.1 < 103 < 100 + 7.1
    # Rate des LKW: Mittel zwischen 0,098 und 0,100 (gegen die Wahrheit 0,10), Standardfehler höchstens 0,001
    rates = [r["rate_lkw"] for r in rows]
    assert ev.num(min(rates), 3) == "0,098" and ev.num(max(rates), 3) == "0,100" and all(r["rate_lkw_se"] < 0.0011 for r in rows)
    assert "naive Rechnung bleibt im Mittel zwischen 38,6 % und 39,3 %" in README and "39,3 % auf 18,4 %" in README
    # die Wahrheit sinkt streng, die Reihe ist streng monoton in h2
    assert all(a["truth"] > b["truth"] for a, b in zip(rows, rows[1:]))


def test_reihe_ueber_den_zeitpunkt_der_frage_der_abstand_waechst_streng():
    rows = DATA["t0_series"]
    assert [r["t0"] for r in rows] == list(C.SWEEP_T0) and all(r["window"] == 40 for r in rows)
    gaps = [r["naive_mean"] - r["truth"] for r in rows]
    assert all(a < b for a, b in zip(gaps, gaps[1:]))
    assert [ev.pp(g) for g in gaps] == ["+0,6 Prozentpunkte", "+2,3 Prozentpunkte", "+4,6 Prozentpunkte", "+10,5 Prozentpunkte", "+20,4 Prozentpunkte", "+26,3 Prozentpunkte"]


def test_reihe_ueber_das_beobachtungsfenster_der_abstand_bleibt():
    rows = DATA["window_series"]
    assert [r["window"] for r in rows] == list(C.WINDOW_OPTIONS)
    assert ev.pct(min(r["naive_mean"] for r in rows)) == "38,7 %" and ev.pct(max(r["naive_mean"] for r in rows)) == "39,4 %"
    assert ev.pct(min(r["censored"] for r in rows)) == "0,4 %" and ev.pct(max(r["censored"] for r in rows)) == "57,1 %"
    assert all(r["naive_mean"] - r["truth"] > 0.09 for r in rows)            # in jeder Zeile über 9 Prozentpunkte über der Wahrheit
    assert "zwischen 38,7 % und 39,4 %" in README and "0,4 % zensierten" in README


def test_reihe_ueber_n_beide_streuen_die_naive_rechnung_mehr():
    rows = DATA["n_series"]
    assert [r["n"] for r in rows] == [50, 100, 300, 600]
    assert all(r["naive_sd"] > r["aj_sd"] for r in rows)
    assert all(a["aj_sd"] > b["aj_sd"] and a["naive_sd"] > b["naive_sd"] for a, b in zip(rows, rows[1:]))   # größeres n, kleinere Streuung
    # Zahlen der Tabelle „Wo die Annahmen enden“ in der App und der Preset-Hilfe: bei n = 50 mehr als doppelt so viel Streuung wie bei n = 300
    small, base = rows[0], rows[2]
    assert small["aj_sd"] > 2 * base["aj_sd"] and small["naive_sd"] > 2 * base["naive_sd"]
    assert [ev.num(100 * x, 1) for x in (small["aj_sd"], base["aj_sd"], small["naive_sd"], base["naive_sd"])] == ["6,7", "2,9", "9,9", "4,2"]
    assert "kumulierte Inzidenz 6,7 gegen 2,9 Prozentpunkte, naive Rechnung 9,9 gegen 4,2" in (ROOT / "app.py").read_text(encoding="utf-8")


def test_summe_der_anteile():
    sm = DATA["sum"]
    assert sm["dataset_n"] == 300 and ev.pct(sm["dataset_censored"]) == "19,3 %" and sm["grid_day"] == 5.0
    assert [ev.pct(sm[k]) for k in ("dataset_naive_lkw", "dataset_naive_bahn", "dataset_remaining", "dataset_naive_sum")] == ["46,2 %", "53,2 %", "25,2 %", "124,6 %"]
    assert [ev.pct(sm[k]) for k in ("dataset_aj_lkw", "dataset_aj_bahn", "dataset_aj_sum")] == ["33,4 %", "41,4 %", "100,0 %"]
    assert sm["runs_naive_sum_above_1"] == 200 and ev.pct(sm["runs_naive_sum_mean"]) == "120,4 %"
    assert ev.pct(sm["runs_naive_sum_min"]) == "115,1 %" and ev.pct(sm["runs_naive_sum_max"]) == "127,6 %"
    assert sm["runs_aj_sum_dev_max"] < 2.1e-15 and sm["dataset_max_abs_dev_aj_sum"] < 2.1e-15
    assert "**124,6 %**" in README and "höchstens um 2·10⁻¹⁵" in README


def test_ungleichung_naiv_nie_unter_kumuliert_in_jeder_messzelle():
    cells = [DATA["standard"]] + DATA["stability"] + DATA["t0_series"] + DATA["window_series"] + DATA["n_series"] + list(DATA["presets"].values())
    assert cells and all(c["naive_below_aj"] == 0 and c["max_naive_minus_aj"] >= 0 for c in cells)


def test_handtabelle_in_der_readme_und_im_json():
    hand = DATA["hand"]["Sechs Container (Handbeispiel)"]
    assert [(r["t"], r["cif1"], r["cif2"], r["naive1"]) for r in hand][-1] == ("8", "7/12", "5/12", "1")
    assert [r["naive1"] for r in hand] == ["1/6", "1/6", "3/8", "3/8", "1"]
    assert "Am Tag 8 steht die kumulierte Inzidenz des LKW bei 7/12 und die der Bahn bei 5/12" in README


# ------------------------------------------------------------------------------------------------ 4. Preset-Hilfen


def test_preset_hilfe_starke_konkurrenz_wahrheit_sinkt_naive_bleibt():
    p, std = DATA["presets"]["Starke Konkurrenz (Bahn 0,30)"], DATA["presets"]["Messung V4 (Standard)"]
    assert p["h2"] == 0.30 == 2 * C.PRESETS["Messung V4 (Standard)"]["h2"]                  # „doppelt so schnell wie in der Standardwelt“
    assert p["truth"] < std["truth"] and abs(p["naive_mean"] - std["naive_mean"]) < 0.01   # die Wahrheit sinkt (21,6 gegen 28,5 %), die naive Rechnung bleibt im Mittel um unter einen Punkt gleich


def test_preset_hilfe_kaum_zensiert_der_fehler_bleibt_und_ohne_konkurrenz_gibt_es_keinen():
    p = DATA["presets"]["Kaum zensiert (Fenster 1000 Tage)"]
    assert p["censored"] < 0.01 and p["naive_mean"] - p["truth"] > 0.10
    assert abs(DATA["presets"]["Ohne Konkurrenz (Bahn-Rate 0)"]["naive_mean"] - DATA["presets"]["Ohne Konkurrenz (Bahn-Rate 0)"]["truth"]) < 0.01   # bei 26 % zensiert


def test_preset_hilfe_spaeter_zeitpunkt_und_kleine_stichprobe():
    late, std = DATA["presets"]["Später Zeitpunkt (Tag 10)"], DATA["presets"]["Messung V4 (Standard)"]
    assert late["t0"] == 10 and (late["naive_mean"] - late["truth"]) > (std["naive_mean"] - std["truth"])
    small = DATA["presets"]["Kleine Stichprobe (n = 50)"]
    assert small["aj_sd"] > 2 * std["aj_sd"] and small["naive_sd"] > 2 * std["naive_sd"] and small["naive_mean"] - small["truth"] > 0.09


def test_ursachenspezifische_rate_bleibt_unter_allen_presets_nahe_der_wahren_rate():
    """Experiment 4 (Zahl in der App): die geschätzte Rate des LKW liegt im Mittel in jedem Preset höchstens 0,002 von der wahren 0,10 entfernt (n = 50: 0,0016)."""
    for name, p in DATA["presets"].items():
        assert abs(p["rate_lkw_mean"] - p["h1"]) < 0.002, name
