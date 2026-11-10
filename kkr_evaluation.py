"""Kennzahlen und Verdikte. Jedes Verdikt ist eine reine Funktion `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`,
tests/test_verdicts.py prüft Schwellen und jede Zahl im Text an handgesetzten Werten (DEMO-PLAYBOOK §12)."""

from fractions import Fraction

import numpy as np

import kkr_constants as C
import kkr_model as M
from kkr_scenario import World, make_sample

# ------------------------------------------------------------------------------------------------ Formatierung (Dezimalkomma)


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma."""
    return f"{100.0 * x:.{digits}f}".replace(".", ",") + " %"


def pp(x: float, digits: int = 1) -> str:
    """Differenz zweier Anteile in Prozentpunkten, mit Vorzeichen."""
    return f"{100.0 * x:+.{digits}f}".replace(".", ",") + " Prozentpunkte"


def pts(x: float, digits: int = 2) -> str:
    """Abstand zweier Anteile (oder Standardfehler eines Anteils) in Prozentpunkten, ohne Vorzeichen."""
    return f"{100.0 * abs(x):.{digits}f}".replace(".", ",") + " Prozentpunkte"


def frac(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# ------------------------------------------------------------------------------------------------ Experiment 3: ein Datensatz über die Zeit


def plot_grid(sample, t0: float, points: int = 241) -> np.ndarray:
    """Zeitraster der Kurven: von 0 bis zum doppelten Zeitpunkt der Frage (mindestens 10 Tage), nicht über die größte beobachtete Zeit hinaus."""
    tmax = min(float(np.max(sample.time)), max(2.0 * t0, 10.0))
    return np.linspace(0.0, tmax, points)


def analyse_curves(world: World):
    """Ein Datensatz (Lauf 0 der Welt) mit allen Kurven; gibt (Datensatz, Kurven) zurück."""
    sample = make_sample(world, 0)
    return sample, M.curves(sample, plot_grid(sample, world.t0))


# ------------------------------------------------------------------------------------------------ Verdikte


def verdict_hand(table: list[dict]) -> tuple[str, str]:
    """Experiment 1: Ergebnis der Handtabelle am Ende (letzte Ereigniszeit)."""
    if not table:
        return "info", "Kein Container wird abgeholt (alle zensiert): Es gibt keine Ereigniszeit und keine Tabelle."
    last = table[-1]
    total = last["cif1"] + last["cif2"] + last["s"]
    if last["cif1"] == last["naive1"]:
        return "success", (
            f"Hier holt die Bahn vor der letzten LKW-Abholung nichts ab, es gibt keine Konkurrenz: Die kumulierte Inzidenz des LKW ist am Ende **{frac(last['cif1'])}** "
            f"≈ {num(float(last['cif1']), 3)}, und 1 − Kaplan–Meier ist {frac(last['naive1'])}: beide stimmen überein."
        )
    return "warning", (
        f"Am Ende steht die kumulierte Inzidenz des LKW bei **{frac(last['cif1'])}** ≈ {num(float(last['cif1']), 3)}, die der Bahn bei {frac(last['cif2'])}; "
        f"LKW {frac(last['cif1'])} + Bahn {frac(last['cif2'])} + noch da {frac(last['s'])} = {frac(total)}. Die naive Rechnung (Bahn als zensiert) kommt auf **{frac(last['naive1'])}** ≈ {num(float(last['naive1']), 3)} "
        f"für den LKW, das sind {num(float(last['naive1'] - last['cif1']), 3)} mehr."
    )


def verdict_study(s: M.Summary, world: World) -> tuple[str, str]:
    """Experiment 2 (Kernfrage): naive Rechnung gegen kumulierte Inzidenz gegen die geschlossene Wahrheit, gezählt über die Läufe."""
    if world.h2 == 0:
        same = s.max_naive_minus_aj < 1e-12 and s.naive_below_aj == 0
        if same:
            return "success", (
                f"Ohne Bahn (Rate 0) gibt es keine Konkurrenz: 1 − Kaplan–Meier und kumulierte Inzidenz sind in allen {s.reps} Läufen gleich "
                f"(größter Unterschied {s.max_naive_minus_aj:.1e}); beide liegen im Mittel bei {pct(s.aj.mean)} gegen die Wahrheit {pct(s.truth)}."
            )
    gap = s.naive.mean - s.truth
    if gap > 3.0 * s.naive.se:
        return "warning", (
            f"Wahrheit {pct(s.truth)}: Die kumulierte Inzidenz liegt im Mittel bei {pct(s.aj.mean)} (Standardfehler {pts(s.aj.se)}), 1 − Kaplan–Meier bei {pct(s.naive.mean)} "
            f"(Standardfehler {pts(s.naive.se)}), also {pp(gap)} über der Wahrheit. In {s.naive_above_truth} von {s.reps} Läufen liegt die naive Rechnung über der Wahrheit, "
            f"die kumulierte Inzidenz in {s.aj_above_truth} von {s.reps}."
        )
    return "info", (
        f"Wahrheit {pct(s.truth)}: kumulierte Inzidenz im Mittel {pct(s.aj.mean)}, 1 − Kaplan–Meier {pct(s.naive.mean)} (Unterschied zur Wahrheit {pp(gap)}, "
        f"Standardfehler {pts(s.naive.se)}). In {s.naive_above_truth} von {s.reps} Läufen liegt die naive Rechnung über der Wahrheit."
    )


def verdict_h2_sweep(rows: list[dict]) -> tuple[str, str]:
    """Experiment 2, zweiter Teil: die Reihe über die Bahn-Rate. Nennt Anfang und Ende der Reihe; ohne Bahn (Rate 0) sind beide Schätzer gleich."""
    first, last = rows[0], rows[-1]
    return "info", (
        f"Von Bahn-Rate {num(first['h2'], 2)} bis {num(last['h2'], 2)} fällt die Wahrheit von {pct(first['truth'])} auf {pct(last['truth'])}; die kumulierte Inzidenz liegt im Mittel "
        f"bei {pct(first['aj'])} bis {pct(last['aj'])}, die naive Rechnung zwischen {pct(min(r['naive'] for r in rows))} und {pct(max(r['naive'] for r in rows))}. "
        f"Mittel über {first['reps']} Läufe je Rate."
    )


def verdict_sum(c: M.Curves, t0: float) -> tuple[str, str]:
    """Experiment 3: die Summe der Anteile an Tag t0, naiv gegen kumulierte Inzidenz."""
    i = int(np.argmin(np.abs(c.grid - t0)))
    day = c.grid[i]
    naive_sum = float(c.sum_naive[i])
    ok_sum = float(c.sum_correct[i])
    if naive_sum > 1.0 + 1e-9:
        return "warning", (
            f"Tag {num(day, 1)}: Naive Anteile LKW {pct(float(c.naive_lkw[i]))} + Bahn {pct(float(c.naive_bahn[i]))} + noch da {pct(float(c.remaining[i]))} = **{pct(naive_sum)}**, "
            f"mehr als 100 %: Das kann kein Anteil derselben Container sein. Kumulierte Inzidenz: {pct(float(c.cif_lkw[i]))} + {pct(float(c.cif_bahn[i]))} + {pct(float(c.remaining[i]))} = {pct(ok_sum)}."
        )
    return "info", (
        f"Tag {num(day, 1)}: Die naive Summe ist {pct(naive_sum)}, die der kumulierten Inzidenz {pct(ok_sum)}. Hier ist kein Überschuss zu sehen, weil bis zu diesem Tag "
        f"höchstens auf einem der beiden Wege abgeholt wurde oder gar nicht."
    )


def verdict_two_questions(rows: list[dict]) -> tuple[str, str]:
    """Experiment 4: die Rate des LKW bleibt bei wachsender Bahn-Rate, der Anteil sinkt."""
    r_min = min(r["rate_lkw"] for r in rows)
    r_max = max(r["rate_lkw"] for r in rows)
    first, last = rows[0], rows[-1]
    return "info", (
        f"Über Bahn-Raten von {num(first['h2'], 2)} bis {num(last['h2'], 2)} liegt die geschätzte Rate des LKW im Mittel zwischen {num(r_min, 3)} und {num(r_max, 3)} je Tag "
        f"(Wahrheit {num(first['rate_true'], 2)}), der Anteil, der per LKW abgeholt wird, sinkt dagegen laut Wahrheit von {pct(first['truth'])} auf {pct(last['truth'])}."
    )
