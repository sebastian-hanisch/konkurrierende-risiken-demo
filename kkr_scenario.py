"""Welt der Daten: zwei Abholwege (LKW, Bahn) mit konstanter Rate, Zensierung durch ein Beobachtungsfenster. Zufall nur über SplitMix64."""

from dataclasses import dataclass

import numpy as np

import kkr_constants as C
import kkr_core as K

LKW, BAHN = 1, 2    # Ursachen-Codes; 0 = zensiert (so verlangt es kkr_core.cumulative_incidence)
CAUSE_NAMES = {0: "noch da (zensiert)", LKW: "LKW", BAHN: "Bahn"}


@dataclass(frozen=True)
class World:
    """Einstellungen einer Welt. Immer mit Schlüsselwörtern aufrufen (ein eingefügtes Feld verschöbe sonst still die Werte)."""

    h1: float = C.DEFAULT_H1          # Abholrate LKW je Tag
    h2: float = C.DEFAULT_H2          # Abholrate Bahn je Tag (0: keine Konkurrenz)
    n: int = C.DEFAULT_N              # Container je Lauf
    window: float = C.DEFAULT_WINDOW  # Obergrenze der Zensierzeit in Tagen
    t0: float = C.DEFAULT_T0          # Zeitpunkt der Frage
    seed: int = C.DEFAULT_SEED        # Zufalls-Seed des ersten Laufs; Lauf r nutzt run_seed(seed, r)


@dataclass(frozen=True)
class Sample:
    """Ein Datensatz: beobachtete Zeit, Ursache (0 zensiert, 1 LKW, 2 Bahn). Die latenten Zeiten bleiben außerhalb (man sieht sie nie)."""

    time: np.ndarray
    cause: np.ndarray


def run_seed(seed: int, r: int) -> int:
    """Zufalls-Seed des Laufs r: feste, voneinander getrennte Ströme (Abstand 1000003 zwischen den Läufen)."""
    return int(seed) + 1000003 * int(r)


def simulate(n: int, h1: float, h2: float, window: float, seed: int) -> Sample:
    """n Container. Gezogen wird immer in derselben Reihenfolge (LKW-Zeit, Bahn-Zeit, Zensierzeit): bei gleichem Zufalls-Seed teilen sich Welten
    mit verschiedener Bahn-Rate dieselben Zufallszahlen für LKW-Zeit und Zensierung, ein Vergleich über h2 ist also gepaart."""
    if n < 1:
        raise K.SurvivalError("Es braucht mindestens einen Container.")
    if h1 < 0 or h2 < 0 or not window > 0:
        raise K.SurvivalError("Die Raten dürfen nicht negativ sein, das Beobachtungsfenster muss größer als 0 sein.")
    rng = K.SplitMix64(seed)
    e1, e2, u = rng.exponential(n), rng.exponential(n), rng.uniform(n)
    t1 = e1 / h1 if h1 > 0 else np.full(n, np.inf)   # Rate 0: der Weg holt nie ab
    t2 = e2 / h2 if h2 > 0 else np.full(n, np.inf)
    cens = window * (1.0 - u)                     # in (0, window]
    first = np.minimum(t1, t2)
    seen = first <= cens
    time = np.where(seen, first, cens)
    cause = np.where(seen, np.where(t1 <= t2, LKW, BAHN), 0).astype(int)
    return Sample(time, cause)


def make_sample(world: World, r: int = 0) -> Sample:
    return simulate(world.n, world.h1, world.h2, world.window, run_seed(world.seed, r))


# --- Handbeispiele (Experiment 1) ------------------------------------------------------------------------------------------------------------------
# Je Muster sechs Container (Zeit in Tagen, Ursache 0 = zensiert, 1 = LKW, 2 = Bahn). Die Werte der Tabellen stehen in tests/test_model.py von Hand gerechnet.
PATTERNS = {
    "Sechs Container (Handbeispiel)": ((2, 1), (3, 2), (4, 1), (5, 0), (6, 2), (8, 1)),
    "Ohne Konkurrenz: die Bahn holt nie ab": ((2, 1), (3, 0), (4, 1), (5, 0), (6, 0), (8, 1)),
    "Gleichzeitig: LKW und Bahn am Tag 4": ((2, 1), (4, 1), (4, 2), (5, 0), (6, 2), (8, 1)),
    "Zensierung vor dem ersten Ereignis": ((1, 0), (3, 1), (4, 2), (5, 0), (6, 1), (8, 2)),
}
DEFAULT_PATTERN = next(iter(PATTERNS))
