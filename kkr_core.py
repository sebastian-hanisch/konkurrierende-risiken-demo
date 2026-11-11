"""Kopie des gemeinsamen Kerns uz_core.py der Überlebenszeit-Linie, Stand 2026-11-10 (nur die Funktionen dieser Demo; der Rest ist gelöscht).

Enthalten: SplitMix64 (Zufall), SurvivalError und check_data (Eingabeprüfung), Kaplan–Meier (für die naive Rechnung „Bahn als zensiert“)
und die kumulierte Inzidenz nach Aalen–Johansen. Nur numpy. Konventionen: Risikomenge bei t = {i : Zeit_i >= t}; bei gleicher Zeit zählt das
Ereignis vor der Zensierung; Ursache 0 = zensiert; Zensierung gilt als unabhängig vom Ereignis. Diese Datei wird nicht verändert (eine
Textkorrektur hier bräche den Vergleich mit der Quelle); Erweiterungen stehen in kkr_model.py.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------------------------------------ Zufall

_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


class SplitMix64:
    """SplitMix64-Strom (Steele, Lea, Flood 2014). Ausgabe k hängt nur von (Seed, k) ab: plattform- und versionsunabhängig.
    Alle Verteilungen sind hier selbst aus Gleichverteilten gebaut (nie numpy-Verteilungen, deren Algorithmen sich zwischen Versionen ändern)."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormal (Box-Muller, nur der Kosinus-Zweig); verbraucht 2·size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)

    def exponential(self, size: int) -> np.ndarray:
        """Exponentialverteilt mit Mittel 1 (Inversion)."""
        return -np.log1p(-self.uniform(size))

    def lognormal(self, size: int, median: float, sigma: float) -> np.ndarray:
        """Lognormal mit gegebenem Median und σ der Log-Skala."""
        return median * np.exp(sigma * self.normal(size))

    def weibull(self, size: int, shape: float, scale: float) -> np.ndarray:
        """Weibull mit Form k und Skala λ, S(t) = exp(−(t/λ)^k) (Inversion)."""
        return scale * self.exponential(size) ** (1.0 / shape)


# ------------------------------------------------------------------------------------------------ Eingabeprüfung


class SurvivalError(ValueError):
    """Ungültige Eingabe (verständliche Meldung für die App)."""


def check_data(time, event, entry=None):
    """Prüft und wandelt in float-/bool-Arrays. Zeiten > 0 und endlich, Ereignis 0/1, Eintritt >= 0 und < Zeit (für jede Zeile)."""
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise SurvivalError("Es braucht mindestens eine Beobachtung (eindimensionale Liste der Zeiten).")
    e_raw = np.asarray(event)
    if e_raw.shape != t.shape:
        raise SurvivalError("Zeiten und Ereignisse müssen gleich viele Einträge haben.")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise SurvivalError("Alle Zeiten müssen endlich und größer als 0 sein.")
    if not np.all(np.isin(e_raw, (0, 1, False, True))):
        raise SurvivalError("Das Ereignis muss 0 (zensiert) oder 1 (Ereignis) sein.")
    e = e_raw.astype(bool)
    if entry is None:
        en = np.zeros_like(t)
    else:
        en = np.asarray(entry, dtype=float)
        if en.shape != t.shape or not np.all(np.isfinite(en)) or np.any(en < 0):
            raise SurvivalError("Der Eintritt muss zu jeder Beobachtung angegeben werden, endlich und nicht negativ sein.")
        bad = np.where(en >= t)[0]
        if bad.size:
            raise SurvivalError(f"Der Eintritt muss vor der Zeit liegen (Beobachtung {int(bad[0]) + 1}: Eintritt {en[bad[0]]:g} ≥ Zeit {t[bad[0]]:g}).")
    return t, e, en

# ------------------------------------------------------------------------------------------------ Kaplan–Meier


def kaplan_meier(time, event, entry=None) -> dict:
    """Kaplan–Meier-Schätzer mit verzögertem Eintritt. Gibt Arrays je Ereigniszeit zurück:
    t, n (im Risiko), d (Ereignisse), s (Überleben nach t), gw (Greenwood-Summe Σ d/(n(n−d)), Terme mit n = d fehlen: dort ist s = 0).
    """
    t, e, en = check_data(time, event, entry)
    ts = np.unique(t[e])
    ts_sorted = np.sort(t)
    en_sorted = np.sort(en)
    n = np.searchsorted(en_sorted, ts, side="left") - np.searchsorted(ts_sorted, ts, side="left")  # #(Eintritt < t) − #(Zeit < t)
    d = np.array([np.sum((t == x) & e) for x in ts], dtype=float) if ts.size else np.zeros(0)
    n = n.astype(float)
    if np.any(n < d):
        raise SurvivalError("Interner Fehler: mehr Ereignisse als Beobachtungen im Risiko.")
    s = np.cumprod(1.0 - d / n) if ts.size else np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(n > d, d / (n * (n - d)), 0.0)
    return {"t": ts, "n": n, "d": d, "s": s, "gw": np.cumsum(terms), "n_obs": int(t.size)}


def _index_at(km: dict, x: float) -> int:
    return int(np.searchsorted(km["t"], x, side="right")) - 1


def survival_at(km: dict, x: float) -> float:
    """Ŝ(x); vor der ersten Ereigniszeit 1."""
    i = _index_at(km, x)
    return 1.0 if i < 0 else float(km["s"][i])


def cumulative_incidence(time, cause, k: int, grid) -> np.ndarray:
    """Kumulierte Inzidenz der Ursache k nach Aalen–Johansen; cause 0 = zensiert, 1, 2, … = Ursachen. Gibt die Werte an den Zeiten in grid zurück."""
    t = np.asarray(time, dtype=float)
    c = np.asarray(cause, dtype=int)
    if t.ndim != 1 or t.size == 0 or c.shape != t.shape or np.any(c < 0) or not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise SurvivalError("Zeiten > 0 und Ursachen 0, 1, 2, … in gleicher Anzahl nötig.")
    ts = np.unique(t[c > 0])
    s_prev, cif = 1.0, 0.0
    times, values = [], []
    for x in ts:
        n = float(np.sum(t >= x))
        dk = float(np.sum((t == x) & (c == k)))
        da = float(np.sum((t == x) & (c > 0)))
        cif += s_prev * dk / n
        s_prev *= 1.0 - da / n
        times.append(x)
        values.append(cif)
    times, values = np.array(times), np.array(values)
    out = []
    for g in np.atleast_1d(grid):
        i = int(np.searchsorted(times, g, side="right")) - 1
        out.append(0.0 if i < 0 else float(values[i]))
    return np.array(out)
