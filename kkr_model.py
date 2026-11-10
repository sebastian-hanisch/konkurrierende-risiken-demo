"""Rechenkern der Demo: geschlossene Wahrheit, Schätzer (naiv und Aalen–Johansen), Handtabelle in Brüchen, Läufe und Reihen.

Jede Funktion ist eine Einheit mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b). Kaplan–Meier und die kumulierte Inzidenz stammen aus
kkr_core.py (Kopie aus der Kernbibliothek der Linie); hier steht nur, was die Demo darüber hinaus braucht."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import kkr_constants as C
import kkr_core as K
from kkr_scenario import BAHN, LKW, Sample, World, make_sample

# ------------------------------------------------------------------------------------------------ geschlossene Wahrheit (konstante Raten)


def truth_cif(h1: float, h2: float, t) -> np.ndarray | float:
    """Anteil, der bis t per Weg 1 abgeholt wird: h1/(h1+h2) · (1 − e^{−(h1+h2)t}). Ohne jede Rate gibt es keine Abholung (0)."""
    h = h1 + h2
    t = np.asarray(t, dtype=float)
    return np.zeros_like(t) if h <= 0 else h1 / h * (-np.expm1(-h * t))


def truth_net(h1: float, t) -> np.ndarray | float:
    """Anteil, der bis t per Weg 1 abgeholt würde, gäbe es den anderen Weg nicht: 1 − e^{−h1·t}."""
    return -np.expm1(-h1 * np.asarray(t, dtype=float))


def truth_remaining(h1: float, h2: float, t) -> np.ndarray | float:
    """Anteil, der zur Zeit t noch nicht abgeholt ist: e^{−(h1+h2)t}."""
    return np.exp(-(h1 + h2) * np.asarray(t, dtype=float))


# ------------------------------------------------------------------------------------------------ Schätzer


@dataclass(frozen=True)
class Estimates:
    """Alle Schätzungen eines Datensatzes zur Zeit t0. naive_*: 1 − Kaplan–Meier, der jeweils andere Weg gilt als zensiert."""

    cif_lkw: float
    cif_bahn: float
    remaining: float      # Kaplan–Meier über beide Wege zusammen: der Anteil, der noch da ist
    naive_lkw: float
    naive_bahn: float


def naive_cif(sample: Sample, k: int, grid) -> np.ndarray:
    """1 − Kaplan–Meier für Weg k, die Abholungen auf dem anderen Weg als zensiert gezählt: die naive Rechnung."""
    km = K.kaplan_meier(sample.time, sample.cause == k)
    return np.array([1.0 - K.survival_at(km, float(g)) for g in np.atleast_1d(grid)])


def remaining_curve(sample: Sample, grid) -> np.ndarray:
    """Kaplan–Meier über beide Wege zusammen (jede Abholung zählt als Ereignis): der Anteil, der noch da ist."""
    km = K.kaplan_meier(sample.time, sample.cause > 0)
    return np.array([K.survival_at(km, float(g)) for g in np.atleast_1d(grid)])


def estimate_at(sample: Sample, t0: float) -> Estimates:
    return Estimates(
        cif_lkw=float(K.cumulative_incidence(sample.time, sample.cause, LKW, [t0])[0]),
        cif_bahn=float(K.cumulative_incidence(sample.time, sample.cause, BAHN, [t0])[0]),
        remaining=float(remaining_curve(sample, [t0])[0]),
        naive_lkw=float(naive_cif(sample, LKW, [t0])[0]),
        naive_bahn=float(naive_cif(sample, BAHN, [t0])[0]),
    )


def estimate_fast(sample: Sample, t0: float) -> Estimates:
    """Dieselben fünf Schätzungen wie `estimate_at`, in einem sortierten Durchgang statt je Schätzer einer Schleife über die Ereigniszeiten (für die
    Läufe in den Experimenten, die hunderte Datensätze auswerten). tests/test_model.py hält beide gegeneinander (auch mit Gleichständen).
    Bei jeder Ereigniszeit x gilt n(x) = #(Zeit >= x); die Faktoren 1 − d/n der drei Produkte (nur LKW, nur Bahn, beide) hängen am selben n."""
    t = np.asarray(sample.time, dtype=float)
    c = np.asarray(sample.cause, dtype=int)
    if t.ndim != 1 or t.size == 0 or c.shape != t.shape or np.any(c < 0) or not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise K.SurvivalError("Die Zeiten müssen endlich und größer als 0 sein, die Ursachen 0, 1 oder 2, beide Listen gleich lang.")
    ts = np.unique(t[c > 0])
    if ts.size == 0:
        return Estimates(0.0, 0.0, 1.0, 0.0, 0.0)
    n = t.size - np.searchsorted(np.sort(t), ts, side="left")        # Zahl im Risiko: Zeit >= x
    d1 = np.bincount(np.searchsorted(ts, t[c == LKW]), minlength=ts.size).astype(float)
    d2 = np.bincount(np.searchsorted(ts, t[c == BAHN]), minlength=ts.size).astype(float)
    n = n.astype(float)
    s_all = np.cumprod(1.0 - (d1 + d2) / n)
    s_prev = np.concatenate([[1.0], s_all[:-1]])
    i = int(np.searchsorted(ts, t0, side="right"))                    # Zahl der Ereigniszeiten <= t0
    if i == 0:
        return Estimates(0.0, 0.0, 1.0, 0.0, 0.0)
    return Estimates(
        cif_lkw=float(np.sum(s_prev[:i] * d1[:i] / n[:i])),
        cif_bahn=float(np.sum(s_prev[:i] * d2[:i] / n[:i])),
        remaining=float(s_all[i - 1]),
        naive_lkw=float(1.0 - np.prod(1.0 - d1[:i] / n[:i])),
        naive_bahn=float(1.0 - np.prod(1.0 - d2[:i] / n[:i])),
    )


def cause_rate(sample: Sample, k: int) -> float:
    """Ursachenspezifische Rate des Wegs k unter konstanter Rate (Exponential-Maximum-Likelihood): Zahl der Abholungen auf Weg k geteilt durch die
    gesamte beobachtete Zeit aller Container (auch der zensierten und der auf dem anderen Weg abgeholten, die bis dahin im Risiko waren)."""
    total = float(np.sum(sample.time))
    if total <= 0:
        raise K.SurvivalError("Die Summe der beobachteten Zeiten ist 0.")
    return float(np.sum(sample.cause == k)) / total


@dataclass(frozen=True)
class Curves:
    """Kurven eines Datensatzes über ein Zeitraster: kumulierte Inzidenz je Weg, noch da, naive Kurven und die Summen."""

    grid: np.ndarray
    cif_lkw: np.ndarray
    cif_bahn: np.ndarray
    remaining: np.ndarray
    naive_lkw: np.ndarray
    naive_bahn: np.ndarray

    @property
    def sum_correct(self) -> np.ndarray:
        """Aalen–Johansen: CIF LKW + CIF Bahn + noch da; ist überall 1."""
        return self.cif_lkw + self.cif_bahn + self.remaining

    @property
    def sum_naive(self) -> np.ndarray:
        """Naive Anteile beider Wege + noch da; übersteigt 1, sobald auf beiden Wegen abgeholt wurde."""
        return self.naive_lkw + self.naive_bahn + self.remaining


def curves(sample: Sample, grid) -> Curves:
    grid = np.asarray(grid, dtype=float)
    return Curves(
        grid=grid,
        cif_lkw=K.cumulative_incidence(sample.time, sample.cause, LKW, grid),
        cif_bahn=K.cumulative_incidence(sample.time, sample.cause, BAHN, grid),
        remaining=remaining_curve(sample, grid),
        naive_lkw=naive_cif(sample, LKW, grid),
        naive_bahn=naive_cif(sample, BAHN, grid),
    )


# ------------------------------------------------------------------------------------------------ Handtabelle in Brüchen


def hand_table(rows) -> list[dict]:
    """Die Tabelle von Hand, in exakten Brüchen: je Ereigniszeit Zahl im Risiko, Abholungen je Weg, Überleben S(t−) → S(t), kumulierte Inzidenz je Weg,
    dazu die naive Rechnung (1 − Kaplan–Meier für den LKW, Bahn als zensiert). rows: Paare (Zeit, Ursache) mit Ursache 0 = zensiert, 1 = LKW, 2 = Bahn.
    Gleichzeitige Abholungen zählen alle zur selben Zeit; ein zensierter Container bleibt bis zu seiner Zeit im Risiko (Ereignis vor Zensierung)."""
    pairs = [(Fraction(t), int(c)) for t, c in rows]
    if not pairs or any(t <= 0 for t, _ in pairs) or any(c not in (0, LKW, BAHN) for _, c in pairs):
        raise K.SurvivalError("Je Container eine Zeit > 0 und eine Ursache 0, 1 oder 2.")
    s, cif1, cif2, km_lkw = Fraction(1), Fraction(0), Fraction(0), Fraction(1)
    out = []
    for x in sorted({t for t, c in pairs if c > 0}):
        n = sum(1 for t, _ in pairs if t >= x)
        d1 = sum(1 for t, c in pairs if t == x and c == LKW)
        d2 = sum(1 for t, c in pairs if t == x and c == BAHN)
        s_prev = s
        cif1 += s_prev * Fraction(d1, n)
        cif2 += s_prev * Fraction(d2, n)
        s = s_prev * (1 - Fraction(d1 + d2, n))
        km_lkw *= 1 - Fraction(d1, n)
        out.append({"t": x, "n": n, "d1": d1, "d2": d2, "s_prev": s_prev, "s": s, "cif1": cif1, "cif2": cif2, "naive1": 1 - km_lkw})
    return out


# ------------------------------------------------------------------------------------------------ Läufe


@dataclass(frozen=True)
class Study:
    """Ergebnisse von `reps` festen Läufen einer Welt, je ein Wert je Lauf."""

    world: World
    cif_lkw: np.ndarray
    cif_bahn: np.ndarray
    remaining: np.ndarray
    naive_lkw: np.ndarray
    naive_bahn: np.ndarray
    rate_lkw: np.ndarray
    rate_bahn: np.ndarray
    censored: np.ndarray   # Anteil zensierter Container je Lauf

    @property
    def reps(self) -> int:
        return int(self.cif_lkw.size)


def run_study(world: World, reps: int = C.REPS) -> Study:
    """`reps` Datensätze mit festen, getrennten Zufalls-Strömen (run_seed); je Datensatz alle Schätzer zur Zeit t0 und die ursachenspezifischen Raten."""
    if reps < 1:
        raise K.SurvivalError("Es braucht mindestens einen Lauf.")
    cols = {k: np.zeros(reps) for k in ("cif_lkw", "cif_bahn", "remaining", "naive_lkw", "naive_bahn", "rate_lkw", "rate_bahn", "censored")}
    for r in range(reps):
        sample = make_sample(world, r)
        est = estimate_fast(sample, world.t0)
        cols["cif_lkw"][r], cols["cif_bahn"][r], cols["remaining"][r] = est.cif_lkw, est.cif_bahn, est.remaining
        cols["naive_lkw"][r], cols["naive_bahn"][r] = est.naive_lkw, est.naive_bahn
        cols["rate_lkw"][r], cols["rate_bahn"][r] = cause_rate(sample, LKW), cause_rate(sample, BAHN)
        cols["censored"][r] = float(np.mean(sample.cause == 0))
    return Study(world=world, **cols)


@dataclass(frozen=True)
class Spread:
    """Mittel, Standardabweichung, Standardfehler des Mittels und 5-%/95-%-Quantil einer Größe über die Läufe."""

    mean: float
    sd: float
    se: float
    q05: float
    q95: float


def spread(values) -> Spread:
    v = np.asarray(values, dtype=float)
    sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    return Spread(float(np.mean(v)), sd, sd / np.sqrt(v.size), float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95)))


@dataclass(frozen=True)
class Summary:
    """Auswertung einer Studie gegen die geschlossene Wahrheit."""

    reps: int
    truth: float              # h1/(h1+h2)·(1 − e^{−(h1+h2)t0})
    truth_net: float          # 1 − e^{−h1·t0}: der Anteil „ohne Bahn“
    aj: Spread
    naive: Spread
    naive_above_truth: int    # Läufe, in denen 1 − KM über der Wahrheit liegt
    aj_above_truth: int
    naive_below_aj: int       # Läufe, in denen 1 − KM unter der kumulierten Inzidenz liegt (nach dem Satz in tests/test_eigenschaften nie)
    max_naive_minus_aj: float
    censored: float          # mittlerer Anteil zensierter Container


def summarize(study: Study) -> Summary:
    w = study.world
    truth = float(truth_cif(w.h1, w.h2, w.t0))
    return Summary(
        reps=study.reps,
        truth=truth,
        truth_net=float(truth_net(w.h1, w.t0)),
        aj=spread(study.cif_lkw),
        naive=spread(study.naive_lkw),
        naive_above_truth=int(np.sum(study.naive_lkw > truth)),
        aj_above_truth=int(np.sum(study.cif_lkw > truth)),
        naive_below_aj=int(np.sum(study.naive_lkw < study.cif_lkw - 1e-12)),
        max_naive_minus_aj=float(np.max(study.naive_lkw - study.cif_lkw)),
        censored=float(np.mean(study.censored)),
    )


def sweep_h2(world: World, h2_values=C.SWEEP_H2, reps: int = C.SWEEP_REPS) -> list[dict]:
    """Dieselbe Welt mit verschiedenen Bahn-Raten (gepaarte Zufallszahlen, siehe simulate). Je Bahn-Rate: Wahrheit, Mittel beider Schätzer,
    Mittel der ursachenspezifischen Rate des LKW mit Standardfehler und die Wahrheit dazu (h1)."""
    rows = []
    for h2 in h2_values:
        study = run_study(World(h1=world.h1, h2=float(h2), n=world.n, window=world.window, t0=world.t0, seed=world.seed), reps)
        s = summarize(study)
        rate = spread(study.rate_lkw)
        rows.append({
            "h2": float(h2), "truth": s.truth, "truth_net": s.truth_net, "aj": s.aj.mean, "naive": s.naive.mean,
            "rate_lkw": rate.mean, "rate_lkw_se": rate.se, "rate_true": world.h1, "rate_bahn": float(np.mean(study.rate_bahn)),
            "naive_above_truth": s.naive_above_truth, "reps": reps,
        })
    return rows
