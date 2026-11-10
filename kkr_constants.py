"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und der Annahmen-Text (App und README)."""

# --- Welt: zwei Abholwege mit konstanter Rate, Zensierung durch ein Beobachtungsfenster -------------------------------------------------------------
# Jeder Container hat zwei latente Zeiten bis zur Abholung (LKW: Exponential mit Rate h1, Bahn: Exponential mit Rate h2, unabhängig); abgeholt wird auf dem Weg
# mit der kleineren Zeit. Zensierzeit: gleichverteilt zwischen 0 und dem Beobachtungsfenster, unabhängig von allem anderen.
H1_MIN, H1_MAX, DEFAULT_H1, H_STEP = 0.02, 0.30, 0.10, 0.01      # Abholrate des LKW je Tag
H2_MIN, H2_MAX, DEFAULT_H2 = 0.0, 0.40, 0.15                     # Abholrate der Bahn je Tag; 0 = es gibt keine Konkurrenz
N_MIN, N_MAX, DEFAULT_N, N_STEP = 50, 600, 300, 50              # Komplexität am Maximum: je Lauf O(Ereigniszeiten · n), mal REPS Läufe
WINDOW_OPTIONS = (5, 10, 15, 25, 40, 1000)                       # Obergrenze der Zensierzeit in Tagen; 1000 = praktisch keine Zensierung
DEFAULT_WINDOW = 15
T0_MIN, T0_MAX, DEFAULT_T0 = 1, 30, 5                            # Zeitpunkt der Frage „Anteil per LKW bis Tag t0“
SEED_MAX, DEFAULT_SEED = 999999, 35

REPS = 200                                # Läufe je Experiment (feste Zufalls-Seeds), in der App und in tools/measure_konkurrierende_risiken.py
SWEEP_H2 = (0.0, 0.05, 0.10, 0.15, 0.25, 0.40)   # Bahn-Raten der Reihe in Experiment 2 und 4
SWEEP_REPS = 200                          # Läufe je Bahn-Rate dieser Reihe (so viele wie REPS)
SWEEP_T0 = (1, 2, 3, 5, 8, 10)            # Zeitpunkte der Messung „Verzerrung wächst mit der Zeit“ (tools/measure_konkurrierende_risiken.py)
Z95 = 1.959963984540054

COLOR_LKW = "#2F6B65"
COLOR_BAHN = "#e08a2b"
COLOR_REST = "rgba(150,150,150,0.55)"
COLOR_NAIVE = "#c0504d"
COLOR_TRUTH = "#7f7f7f"
COLOR_LINE = "#4c78a8"

# --- Presets ------------------------------------------------------------------------------------------------------------------------------------
# Schlüssel: h1, h2, n, window, t0, seed (Welt und Frage); fehlt ein Schlüssel, gilt der Standardwert.
PRESETS = {
    "Messung V4 (Standard)": {"h1": 0.10, "h2": 0.15, "n": 300, "window": 15, "t0": 5, "seed": DEFAULT_SEED},
    "Ohne Konkurrenz (Bahn-Rate 0)": {"h1": 0.10, "h2": 0.0, "n": 300, "window": 15, "t0": 5, "seed": DEFAULT_SEED},
    "Starke Konkurrenz (Bahn 0,30)": {"h1": 0.10, "h2": 0.30, "n": 300, "window": 15, "t0": 5, "seed": DEFAULT_SEED},
    "Kaum zensiert (Fenster 1000 Tage)": {"h1": 0.10, "h2": 0.15, "n": 300, "window": 1000, "t0": 5, "seed": DEFAULT_SEED},
    "Später Zeitpunkt (Tag 10)": {"h1": 0.10, "h2": 0.15, "n": 300, "window": 40, "t0": 10, "seed": DEFAULT_SEED},
    "Kleine Stichprobe (n = 50)": {"h1": 0.10, "h2": 0.15, "n": 50, "window": 15, "t0": 5, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Messung V4 (Standard)": "LKW-Rate 0,10 und Bahn-Rate 0,15 je Tag, 300 Container, Beobachtungsfenster 15 Tage, Frage: Anteil per LKW bis Tag 5. Das ist die Welt der Messung dieser Demo.",
    "Ohne Konkurrenz (Bahn-Rate 0)": "Es gibt nur den LKW. Dann ist 1 − Kaplan–Meier dasselbe wie die kumulierte Inzidenz: kein Unterschied zwischen den Schätzern.",
    "Starke Konkurrenz (Bahn 0,30)": "Die Bahn holt doppelt so schnell ab wie in der Standardwelt. Die Wahrheit sinkt, die naive Rechnung weiß davon nichts.",
    "Kaum zensiert (Fenster 1000 Tage)": "Fast jeder Container wird bis zur Abholung beobachtet. Der Fehler der naiven Rechnung bleibt: er kommt von der Konkurrenz, nicht von der Zensierung.",
    "Später Zeitpunkt (Tag 10)": "Frage an Tag 10 statt Tag 5, Beobachtungsfenster 40 Tage (damit bei Tag 10 noch Container beobachtet werden). Je später die Frage, desto größer der Abstand.",
    "Kleine Stichprobe (n = 50)": "50 Container: die Standardabweichung über die Läufe ist mehr als doppelt so groß wie bei 300; der Abstand der naiven Rechnung zur Wahrheit bleibt.",
}

# --- Annahmen, Grenzen und bekannte Probleme (derselbe Text steht in der README; ein Test vergleicht beide) -------------------------------------
ASSUMPTIONS_MARKDOWN = """\
**Annahmen**
- Die Zensierung ist unabhängig vom Ereignis: Ein zensierter Container hätte dieselbe Abholrate wie die, die noch im Risiko bleiben (hier erzeugt durch ein Beobachtungsfenster mit gleichverteilter Zensierzeit, unabhängig von allem anderen).
- Jeder Container wird auf genau einem Weg abgeholt (LKW oder Bahn), und die Ursache ist bei jeder beobachteten Abholung bekannt.
- Konstante Abholraten je Weg (Exponentialverteilung). Nur dadurch gibt es die geschlossene Formel, gegen die die Schätzer in der Demo gehalten werden; die Schätzer selbst brauchen die Konstanz nicht.
- Die naive Rechnung (Bahn als zensiert) beantwortet unter unabhängigen Abholzeiten die Frage, wie viele Container per LKW abgeholt würden, wenn es die Bahn nicht gäbe (Lesart und Eigenmessung in dieser Welt, nicht aus einer Quelle). Die kumulierte Inzidenz beantwortet die Frage nach dem Anteil, der tatsächlich per LKW abgeholt wird.

**Grenzen**
- Alle Zahlen stammen aus einer simulierten Welt mit zwei Ursachen und konstanten Raten; andere Welten (mehr Ursachen, steigende Raten) sind nicht gemessen; Größen und Richtung können dort anders sein.
- Ob die latenten Abholzeiten unabhängig sind, lässt sich aus (Zeit, Ursache) allein weder bestätigen noch widerlegen (Tsiatis 1975, im Original gelesen). Die kumulierte Inzidenz braucht diese Unabhängigkeit nicht, die Lesart „ohne Bahn“ der naiven Rechnung braucht sie.
- Fine–Gray (Regression auf die Rate der Teilverteilung) kommt nur als Satz vor und ist nicht gemessen.
- Es gibt zwei Ursachen und keine Merkmale (keine Regression, keine Gruppenvergleiche).

**Bekannte Probleme und Unsicherheiten**
- Die Läufe in Experiment 2 und 4 sind Stichproben mit festen Zufalls-Seeds: Zahlen wie „in 200 von 200 Läufen“ gelten für diese Seeds, andere Seeds ändern einzelne Zählungen.
- Am Ende des Beobachtungsfensters sind kaum noch Container im Risiko; die Kurven sind dort grob.
- Die Seite ist im Browser nur am Rechner geprüft (Seitenaufruf, Regler über AppTest), nicht auf Mobilgeräten.
"""
