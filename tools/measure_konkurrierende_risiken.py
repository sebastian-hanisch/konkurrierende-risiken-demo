"""Messskript der Demo: jede Zahl der README steht in `tools/measure_konkurrierende_risiken_result.json` (eingefroren, feste Zufalls-Seeds).

Aufruf (Repo-Wurzel):  _venvs/test/Scripts/python.exe -X utf8 tools/measure_konkurrierende_risiken.py            schreibt die JSON-Datei
                       _venvs/test/Scripts/python.exe -X utf8 tools/measure_konkurrierende_risiken.py --print    zeigt die Tabellen, schreibt nichts

Alle Messungen laufen mit dem Code der Demo (SplitMix64, kkr_model); Läufe je Zelle: C.REPS = 200, Lauf r hat den Zufalls-Seed `run_seed(Zufalls-Seed, r)`.
Die Funktionen sind rein; tests/test_claims.py rechnet Teile davon neu und prüft, dass die Tabellen der README wörtlich aus der Datei stammen.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kkr_constants as C  # noqa: E402
import kkr_core as K  # noqa: E402
import kkr_evaluation as ev  # noqa: E402
import kkr_model as M  # noqa: E402
from kkr_scenario import PATTERNS, World, make_sample  # noqa: E402

RESULT = Path(__file__).with_name("measure_konkurrierende_risiken_result.json")
STANDARD = World()                                   # LKW 0,10, Bahn 0,15, n = 300, Fenster 15 Tage, Tag 5, Zufalls-Seed 35
SEEDS_STABILITY = (35, 101, 202, 303, 404, 505)      # sechs verschiedene Zufalls-Seeds für die Stabilitätsprobe
WINDOW_SERIES = C.WINDOW_OPTIONS
N_SERIES = (50, 100, 300, 600)


def cell(world: World, reps: int = C.REPS) -> dict:
    """Eine Zelle: alle Kennzahlen der Läufe als Zahlen (JSON-fähig)."""
    study = M.run_study(world, reps)
    s = M.summarize(study)
    return {
        "h1": world.h1, "h2": world.h2, "n": world.n, "window": world.window, "t0": world.t0, "seed": world.seed, "reps": s.reps,
        "truth": s.truth, "truth_net": s.truth_net, "censored": s.censored,
        "aj_mean": s.aj.mean, "aj_se": s.aj.se, "aj_sd": s.aj.sd, "aj_q05": s.aj.q05, "aj_q95": s.aj.q95,
        "naive_mean": s.naive.mean, "naive_se": s.naive.se, "naive_sd": s.naive.sd, "naive_q05": s.naive.q05, "naive_q95": s.naive.q95,
        "naive_above_truth": s.naive_above_truth, "aj_above_truth": s.aj_above_truth, "naive_below_aj": s.naive_below_aj,
        "max_naive_minus_aj": s.max_naive_minus_aj,
        "rate_lkw_mean": float(np.mean(study.rate_lkw)), "rate_lkw_se": M.spread(study.rate_lkw).se,
        "rate_bahn_mean": float(np.mean(study.rate_bahn)),
    }


def sum_cell(world: World, reps: int = C.REPS) -> dict:
    """Experiment 3: die Summe der Anteile. Ein Datensatz (Lauf 0) über das Raster und die Summe an Tag t0 in allen Läufen."""
    sample, c = ev.analyse_curves(world)
    i = int(np.argmin(np.abs(c.grid - world.t0)))
    naive_sums, aj_sums = [], []
    for r in range(reps):
        e = M.estimate_fast(make_sample(world, r), world.t0)
        naive_sums.append(e.naive_lkw + e.naive_bahn + e.remaining)
        aj_sums.append(e.cif_lkw + e.cif_bahn + e.remaining)
    naive_sums, aj_sums = np.array(naive_sums), np.array(aj_sums)
    return {
        "grid_day": float(c.grid[i]), "dataset_naive_lkw": float(c.naive_lkw[i]), "dataset_naive_bahn": float(c.naive_bahn[i]),
        "dataset_remaining": float(c.remaining[i]), "dataset_naive_sum": float(c.sum_naive[i]),
        "dataset_aj_lkw": float(c.cif_lkw[i]), "dataset_aj_bahn": float(c.cif_bahn[i]), "dataset_aj_sum": float(c.sum_correct[i]),
        "dataset_max_abs_dev_aj_sum": float(np.max(np.abs(c.sum_correct - 1.0))),
        "runs_naive_sum_above_1": int(np.sum(naive_sums > 1.0 + 1e-9)), "runs_aj_sum_dev_max": float(np.max(np.abs(aj_sums - 1.0))),
        "runs_naive_sum_mean": float(np.mean(naive_sums)), "runs_naive_sum_min": float(np.min(naive_sums)), "runs_naive_sum_max": float(np.max(naive_sums)),
        "reps": reps, "dataset_censored": float(np.mean(sample.cause == 0)), "dataset_n": int(sample.time.size),
    }


def hand_numbers() -> dict:
    """Die Handtabellen aller Muster (kein Zufall) als Brüche in Textform."""
    out = {}
    for name, rows in PATTERNS.items():
        t = M.hand_table(rows)
        out[name] = [{"t": str(r["t"]), "n": r["n"], "d1": r["d1"], "d2": r["d2"], "s_prev": str(r["s_prev"]), "s": str(r["s"]),
                      "cif1": str(r["cif1"]), "cif2": str(r["cif2"]), "naive1": str(r["naive1"])} for r in t]
    return out


def measure() -> dict:
    sweep = M.sweep_h2(STANDARD)
    return {
        "standard": cell(STANDARD),
        "stability": [cell(World(seed=s)) for s in SEEDS_STABILITY],
        "sweep_h2": sweep,
        "sum": sum_cell(STANDARD),
        "t0_series": [cell(World(window=40, t0=t)) for t in C.SWEEP_T0],
        "window_series": [cell(World(window=w)) for w in WINDOW_SERIES],
        "n_series": [cell(World(n=n)) for n in N_SERIES],
        "presets": {name: cell(World(**{k: v for k, v in p.items()})) for name, p in C.PRESETS.items()},
        "hand": hand_numbers(),
    }


# ------------------------------------------------------------------------------------------------ Tabellen der README (aus der JSON-Datei)


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(headers, rows) -> str:
    return "\n".join([_row(headers), "|" + "|".join("---" for _ in headers) + "|"] + [_row(r) for r in rows])


def tables(data: dict) -> dict[str, str]:
    """Die Tabellen der README, wörtlich aus den eingefrorenen Zahlen gebildet (tests/test_claims.py sucht sie in der README)."""
    out = {}
    p = ev.pct
    out["sweep_h2"] = _table(
        ["Bahn-Rate", "Wahrheit", "Kumulierte Inzidenz (Mittel)", "1 − Kaplan–Meier (Mittel)", "naive Rechnung über der Wahrheit", "Rate des LKW, geschätzt (Mittel)"],
        [[ev.num(r["h2"], 2), p(r["truth"]), p(r["aj"]), p(r["naive"]), f"{r['naive_above_truth']} von {r['reps']}", ev.num(r["rate_lkw"], 3)] for r in data["sweep_h2"]],
    )
    out["t0_series"] = _table(
        ["Frage an Tag", "Wahrheit", "Kumulierte Inzidenz (Mittel)", "1 − Kaplan–Meier (Mittel)", "Abstand der naiven Rechnung zur Wahrheit"],
        [[int(r["t0"]), p(r["truth"]), p(r["aj_mean"]), p(r["naive_mean"]), ev.pp(r["naive_mean"] - r["truth"])] for r in data["t0_series"]],
    )
    out["window_series"] = _table(
        ["Beobachtungsfenster (Tage)", "zensiert (Mittel)", "Wahrheit", "Kumulierte Inzidenz (Mittel)", "1 − Kaplan–Meier (Mittel)"],
        [[int(r["window"]), p(r["censored"]), p(r["truth"]), p(r["aj_mean"]), p(r["naive_mean"])] for r in data["window_series"]],
    )
    out["n_series"] = _table(
        ["Container n", "Standardabweichung der kumulierten Inzidenz", "Standardabweichung von 1 − Kaplan–Meier", "naive Rechnung über der Wahrheit"],
        [[int(r["n"]), ev.pts(r["aj_sd"]), ev.pts(r["naive_sd"]), f"{r['naive_above_truth']} von {r['reps']}"] for r in data["n_series"]],
    )
    out["presets"] = _table(
        ["Preset", "Wahrheit", "Kumulierte Inzidenz (Mittel)", "1 − Kaplan–Meier (Mittel)", "naive Rechnung über der Wahrheit"],
        [[name, p(r["truth"]), p(r["aj_mean"]), p(r["naive_mean"]), f"{r['naive_above_truth']} von {r['reps']}"] for name, r in data["presets"].items()],
    )
    out["hand"] = _table(
        ["Tag t", "im Risiko", "LKW", "Bahn", "S(t−) → S(t)", "kumulierte Inzidenz LKW", "kumulierte Inzidenz Bahn", "naiv: 1 − Kaplan–Meier (LKW)"],
        [[r["t"], r["n"], r["d1"], r["d2"], f"{r['s_prev']} → {r['s']}", r["cif1"], r["cif2"], r["naive1"]] for r in data["hand"][next(iter(data["hand"]))]],
    )
    return out


def main() -> None:
    data = measure()
    if "--print" in sys.argv:
        for name, table in tables(data).items():
            print(f"\n== {name}\n{table}")
        print("\nstandard:", json.dumps(data["standard"], indent=1))
        print("sum:", json.dumps(data["sum"], indent=1))
        return
    RESULT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("geschrieben:", RESULT)


if __name__ == "__main__":
    main()
