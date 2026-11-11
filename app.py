"""Konkurrierende Risiken - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Container im Hafen wird entweder per LKW oder per Bahn abgeholt; wer zuerst kommt, nimmt dem anderen die Chance. Vier Experimente: die kumulierte Inzidenz
von Hand (Brüche), die naive Rechnung (Bahn als zensiert) gegen die kumulierte Inzidenz in einer Simulation, die Summe der Anteile, und die zwei Fragen
„Anteil“ und „Rate“ aus denselben Daten.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kkr_constants as C
import kkr_core as K
import kkr_evaluation as ev
import kkr_model as M
import kkr_visualization as V
from kkr_layout import dunkles_design, kopf_css
from kkr_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kkr_scenario import DEFAULT_PATTERN, PATTERNS, World

st.set_page_config(page_title="Konkurrierende Risiken – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _study(world: World):
    study = M.run_study(world, C.REPS)
    return study, M.summarize(study)


@st.cache_data(show_spinner=False)
def _sweep(world: World):
    """Die Reihe über die Bahn-Rate hängt nicht an der Bahn-Rate der Seitenleiste: world kommt mit h2 = 0 an, damit der Cache bei h2-Änderungen trifft."""
    return M.sweep_h2(world, C.SWEEP_H2, C.SWEEP_REPS)


@st.cache_data(show_spinner=False)
def _curves(world: World):
    sample, curves = ev.analyse_curves(world)
    return sample, curves


def md_table(headers, rows) -> str:
    """Markdown-Tabelle aus Texten (kein | in den Zellen)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kkr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🚚 Konkurrierende Risiken: LKW oder Bahn")
st.markdown(
    """
Ein Container im Hafen wird auf **einem von zwei Wegen** abgeholt, per **LKW** oder per **Bahn**; wer zuerst kommt, nimmt dem anderen die Chance. Gefragt ist der **Anteil**, der bis Tag $t$ **per LKW**
abgeholt wird. Kaplan–Meier kennt nur „Ereignis oder noch da“. Wer die Bahn-Abholungen als **zensiert** behandelt und 1 − Kaplan–Meier für den LKW rechnet, beantwortet unter unabhängigen Abholzeiten eine andere Frage
(wie viele würden per LKW abgeholt, **gäbe es die Bahn nicht**); der Anteil, der wirklich per LKW geht, ist kleiner, sobald die Bahn überhaupt abholt. Die passende Rechnung für diese Frage ist die **kumulierte Inzidenz** (Aalen–Johansen).

**Annahme in dieser Demo: Die Zensierung ist unabhängig vom Ereignis** (ein zensierter Container hätte dieselbe Abholrate wie ein gleicher Container, der noch im Risiko bleibt); die Welt unten
erfüllt sie, geprüft wird sie hier nicht. Die Raten sind ausgedachte Zahlen, keine Zahlen aus echten Terminals.

Weiter unten: **🔬 Experiment 1** (die Tabelle von Hand, in Brüchen), **❓ Kernfrage mit Experiment 2** (naiv gegen richtig), **🔬 Experiment 3** (die Summe der Anteile), **🔬 Experiment 4** (zwei Fragen,
zwei Antworten), **🚧 Wo die Annahmen enden** sowie die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die kumulierte Inzidenz", expanded=True):
    st.markdown(
        """
1. **Wer noch wartet.** $S(t)$ ist der Anteil der Container, die zur Zeit $t$ noch auf keinem der beiden Wege abgeholt sind. Er sinkt bei **jeder** Abholung, gleich auf welchem Weg.
2. **Anteil einer Abholzeit.** Holt der LKW zur Zeit $t$ genau $d_{\\text{LKW}}$ von $n$ im Risiko stehenden Containern ab, geht davon der Teil $S(t^-)\\cdot d_{\\text{LKW}}/n$ aller Container per LKW weg
   (nicht nur $d_{\\text{LKW}}/n$ der noch Wartenden: Die Wartenden sind nur noch der Anteil $S(t^-)$ aller Container).
3. **Aufsummieren.** Die **kumulierte Inzidenz** des LKW ist die Summe dieser Anteile über alle Abholzeiten bis $t$; die der Bahn entsprechend. Zusammen mit $S(t)$ ergibt sich immer 1.
4. **Zensierte Container** zählen nie als Abholung, bleiben aber bis zu ihrer Zeit im Risiko.
5. **Die naive Rechnung** nimmt statt $S$ ein Produkt, das nur die LKW-Abholungen kennt. Die Bahn-Container gelten als „zensiert“, scheinen also weiter für den LKW im Spiel zu bleiben; das Produkt ist zu groß, und der Anteil wird zu hoch.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt der Daten** (Experimente 2 bis 4)")
    h1 = st.slider("Abholrate des LKW je Tag", *bounds("h1_slider"), step=C.H_STEP, key="h1_slider",
                   help="Konstante Rate: Ein noch wartender Container wird je Tag mit dieser Rate vom LKW abgeholt (mittlere Wartezeit allein beim LKW: 1 durch die Rate).")
    h2 = st.slider("Abholrate der Bahn je Tag", *bounds("h2_slider"), step=C.H_STEP, key="h2_slider",
                   help="Konstante Rate der Bahn. Bei 0 holt die Bahn nie ab: dann gibt es keine Konkurrenz.")
    n = st.slider("Container n je Lauf", *bounds("n_slider"), step=C.N_STEP, key="n_slider")
    window = st.select_slider("Beobachtungsfenster (Tage)", options=list(C.WINDOW_OPTIONS), key="window_select",
                              help="Obergrenze der gleichverteilten Zensierzeit. Je kürzer das Fenster, desto mehr Container sind bei Fensterende noch nicht abgeholt (zensiert); 1000 Tage heißt praktisch keine Zensierung.")
    st.markdown("**Frage**")
    t0 = st.slider("Zeitpunkt der Frage: bis Tag", *bounds("t0_slider"), key="t0_slider",
                   help="Gefragt wird nach dem Anteil, der bis zu diesem Tag per LKW abgeholt wurde. Liegt der Tag hinter dem Beobachtungsfenster, sind dort kaum noch Container im Risiko.")
    seed = st.number_input("Zufalls-Seed der Daten", *bounds("seed_input"), key="seed_input", step=1,
                           help="Jeder der Läufe in Experiment 2 und 4 hat einen eigenen, festen Zufallsstrom, der aus diesem Zufalls-Seed abgeleitet ist.")
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"h1_slider": float(h1), "h2_slider": float(h2), "n_slider": int(n), "window_select": int(window), "t0_slider": int(t0), "seed_input": int(seed)})
world = World(h1=float(h1), h2=float(h2), n=int(n), window=float(window), t0=float(t0), seed=int(seed))

# --- Experiment 1: die Tabelle von Hand -----------------------------------------------------------------------------------------------------------

st.markdown("## 🧮 Das Verfahren von Hand: sechs Container")
st.subheader("🔬 Experiment 1: Die Tabelle in Brüchen")
st.markdown(
    """
Sechs Container werden zu den Tagen der Tabelle abgeholt (LKW oder Bahn) oder sind bei Fensterende noch da (**zensiert**). Bei jeder Abholzeit rechnet die Tabelle: **im Risiko** sind alle mit Zeit $\\ge t$,
$S$ sinkt um den Anteil aller Abholungen, und die **kumulierte Inzidenz** eines Wegs wächst um $S(t^-)\\cdot d/n$. Die letzte Spalte rechnet naiv: Bahn-Container zensiert, 1 − Kaplan–Meier für den LKW.
Wähle ein Muster und wie viele Zeilen du sehen willst.
"""
)
c1, c2 = st.columns(2)
pattern = c1.selectbox("Muster der sechs Container", list(PATTERNS), key="pattern_select",
                       help="Jedes Muster sind sechs Paare (Tag, Weg). Die Muster zeigen den Normalfall, einen Fall ohne Konkurrenz, gleichzeitige Abholungen und Zensierung vor der ersten Abholung.")
rows_all = PATTERNS.get(pattern, PATTERNS[DEFAULT_PATTERN])
table = M.hand_table(rows_all)
st.markdown("**Die Container (Tag, Weg)**: " + ", ".join(f"({t:g}, {'LKW' if c == 1 else 'Bahn' if c == 2 else 'noch da'})" for t, c in rows_all))
shown = c2.slider("Zeilen der Tabelle zeigen", 1, len(table), len(table), key=f"rows_slider_{list(PATTERNS).index(pattern)}",
                  help="Die Tabelle wächst Zeile für Zeile; das Verdikt rechnet immer mit allen Zeilen, die senkrechte Linie im Diagramm steht bei der letzten gezeigten Zeile.") if len(table) > 1 else 1
shown_rows = table[:shown]
st.markdown(md_table(
    ["Tag t", "im Risiko", "LKW", "Bahn", "S(t−) → S(t)", "kumulierte Inzidenz LKW", "kumulierte Inzidenz Bahn", "naiv: 1 − Kaplan–Meier (LKW)"],
    [[ev.frac(r["t"]), r["n"], r["d1"], r["d2"], f"{ev.frac(r['s_prev'])} → {ev.frac(r['s'])}", ev.frac(r["cif1"]), ev.frac(r["cif2"]), ev.frac(r["naive1"])]
     for r in shown_rows],
))
st.plotly_chart(V.build_hand(table, float(shown_rows[-1]["t"]) if shown_rows else None), width="stretch", key="hand_chart")
kind, text = ev.verdict_hand(table)
getattr(st, kind)(text)

st.markdown("---")

# --- Kernfrage und Experiment 2: naiv gegen richtig -----------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viel Prozent der Container werden bis Tag t per LKW abgeholt, und wie weit führt die naive Rechnung in die Irre?")
st.subheader("🔬 Experiment 2: Naive Rechnung gegen kumulierte Inzidenz")
st.markdown(
    f"""
Welt: jeder Container hat eine LKW-Zeit (Exponential mit der Rate der Seitenleiste) und eine Bahn-Zeit (Rate der Bahn), beide unabhängig; abgeholt wird auf dem Weg mit der kleineren Zeit. Er wird bis zur Abholung oder
bis zum Ende des Beobachtungsfensters beobachtet (Zensierzeit gleichverteilt zwischen 0 und dem Fenster, unabhängig von allem anderen). Für konstante Raten gibt es die **geschlossene Wahrheit**
$\\text{{CIF}}_{{\\text{{LKW}}}}(t) = \\frac{{h_1}}{{h_1 + h_2}}\\,(1 - e^{{-(h_1 + h_2)\\,t}})$. Es laufen **{C.REPS} Läufe** mit festen Zufalls-Seeds; jeder zieht $n$ Container neu und schätzt den Anteil bis Tag $t$ zweimal:
naiv (Bahn zensiert) und mit der kumulierten Inzidenz.
"""
)
try:
    study, summary = _study(world)
except K.SurvivalError as err:
    st.error(f"Die Rechnung ist nicht möglich: {err}")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Wahrheit", ev.pct(summary.truth), help="Geschlossene Formel h1/(h1+h2) · (1 − e^(−(h1+h2)·t)) bei den Raten der Seitenleiste.")
k2.metric("Kumulierte Inzidenz", ev.pct(summary.aj.mean), help="Mittel über die Läufe.")
k3.metric("1 − Kaplan–Meier", ev.pct(summary.naive.mean), help="Mittel über die Läufe; die Bahn-Abholungen gelten als zensiert.")
k4.metric("Zensiert", ev.pct(summary.censored), help="Mittlerer Anteil der Container, die bei Fensterende noch nicht abgeholt sind.")
st.caption(f"Mittel über {summary.reps} Läufe. Standardfehler des Mittels: kumulierte Inzidenz {ev.pts(summary.aj.se)}, 1 − Kaplan–Meier {ev.pts(summary.naive.se)}.")
st.plotly_chart(V.build_estimators(study, summary), width="stretch", key="estimators_chart")
st.caption(
    f"Je Lauf ein Punkt, Kasten: mittlere Hälfte der Läufe, gestrichelte Linie im Kasten: Mittel. Durchgezogen: die Wahrheit. Gepunktet: der Anteil, der per LKW abgeholt würde, gäbe es die Bahn nicht "
    f"({ev.pct(summary.truth_net)} = 1 − e^(−h₁·t)); die naive Rechnung liegt im Mittel bei {ev.pct(summary.naive.mean)}."
)
kind, text = ev.verdict_study(summary, world)
getattr(st, kind)(text)

st.markdown("**Wie ändert sich das mit der Abholrate der Bahn?** Dieselbe Welt mit verschiedenen Bahn-Raten (gleiche Zufallszahlen für LKW-Zeit und Zensierung):")
try:
    sweep = _sweep(World(h1=world.h1, h2=0.0, n=world.n, window=world.window, t0=world.t0, seed=world.seed))
except K.SurvivalError as err:
    st.error(f"Die Reihe ist nicht möglich: {err}")
    st.stop()
st.plotly_chart(V.build_h2_sweep(sweep), width="stretch", key="sweep_chart")
st.caption(f"Mittel über {C.SWEEP_REPS} Läufe je Bahn-Rate (feste Zufalls-Seeds), die übrigen Einstellungen wie in der Seitenleiste.")
kind, text = ev.verdict_h2_sweep(sweep)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3: die Summe stimmt ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Die Summe stimmt")
st.markdown(
    """
Jeder Container wird irgendwann auf einem der beiden Wege abgeholt oder ist noch da. Die drei Anteile **LKW**, **Bahn** und **noch da** müssen deshalb zu jeder Zeit zusammen **100 %** ergeben.
Hier ein Datensatz (Lauf 1 der Welt oben): links die Anteile nach der kumulierten Inzidenz, rechts die Summe, einmal mit den naiven Anteilen beider Wege (jeweils der andere Weg zensiert) und einmal mit der kumulierten Inzidenz.
"""
)
try:
    sample, curves = _curves(world)
except K.SurvivalError as err:
    st.error(f"Der Datensatz ist nicht auswertbar: {err}")
    st.stop()
st.plotly_chart(V.build_sum(curves, world), width="stretch", key="sum_chart")
st.caption(f"Ein Datensatz mit {sample.time.size} Containern, {ev.pct(float((sample.cause == 0).mean()))} davon zensiert. Die kumulierte Inzidenz beider Wege und der Anteil, der noch da ist, ergeben in diesem Datensatz bis auf Rechenrauschen überall 1.")
kind, text = ev.verdict_sum(curves, world.t0)
getattr(st, kind)(text)
st.plotly_chart(V.build_curves(curves, world), width="stretch", key="curves_chart")
st.caption("Derselbe Datensatz, nur der LKW: die naive Kurve liegt nie unter der kumulierten Inzidenz (Beweis im Expander „📐 Mathematische Formulierung“); grau durchgezogen: die Wahrheit, grau gepunktet: der Anteil ohne Bahn.")

st.markdown("---")

# --- Experiment 4: zwei Fragen, zwei Antworten -----------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Zwei Fragen, zwei Antworten")
st.markdown(
    f"""
Aus denselben Daten lassen sich zwei verschiedene Fragen beantworten.
**Frage 1: Wie wahrscheinlich geht ein Container per LKW?** Antwort: die kumulierte Inzidenz. Sie hängt an **beiden** Raten, denn jeder Container, den die Bahn nimmt, fällt für den LKW weg.
**Frage 2: Wie schnell holt der LKW die noch wartenden Container ab?** Antwort: die **ursachenspezifische Rate** des LKW, hier unter konstanter Rate die Zahl der LKW-Abholungen geteilt durch die gesamte
beobachtete Zeit aller Container. Sie bezieht sich nur auf die, die noch warten; in dieser Welt (unabhängige Abholzeiten, konstante Raten) geht die Bahn-Rate nicht in sie ein. Die Reihe über die Bahn-Rate ({C.SWEEP_REPS} Läufe je Wert) zeigt beides nebeneinander.
"""
)
st.plotly_chart(V.build_two_questions(sweep, world.t0), width="stretch", key="two_questions_chart")
st.caption("Rechts: Mittel der geschätzten Rate über die Läufe, Fehlerbalken ± 1 Standardfehler des Mittels.")
kind, text = ev.verdict_two_questions(sweep)
getattr(st, kind)(text)
st.markdown(
    "Ein dritter Ansatz, die Regression auf die Rate der Teilverteilung (Fine und Gray 1999, aus dem Gedächtnis genannt, nicht eingesehen), verbindet beide Fragen mit Merkmalen der Container; "
    "diese Demo rechnet und misst ihn nicht."
)

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was gilt, wenn sie verletzt ist |
|---|---|
| **Zensierung unabhängig vom Ereignis** | Nicht gemessen: Die Welt dieser Demo erzeugt unabhängige Zensierung, deshalb sagt keine Zahl hier etwas über abhängige Zensierung. |
| **Die Ursache jeder beobachteten Abholung ist bekannt** | Nicht gemessen: Falsch zugeordnete oder fehlende Ursachen verschieben die kumulierte Inzidenz; die Demo erzeugt keine. |
| **Unabhängige latente Abholzeiten** (Lesart der naiven Zahl als „ohne Bahn“) | Aus (Zeit, Ursache) allein weder zu bestätigen noch zu widerlegen (Tsiatis 1975, im Original gelesen). Die kumulierte Inzidenz braucht sie nicht, die Lesart „ohne Bahn“ schon. |
| **Konstante Raten** | Die geschlossene Wahrheit und die Rate in Experiment 4 setzen sie voraus; die kumulierte Inzidenz selbst nicht. Steigende oder fallende Raten sind hier nicht gemessen. |
| **Genug Container im Risiko** | Gemessen (Reihe über n in der README): Bei n = 50 ist die Standardabweichung über die Läufe mehr als doppelt so groß wie bei n = 300 (kumulierte Inzidenz 6,7 gegen 2,9 Prozentpunkte, naive Rechnung 9,9 gegen 4,2). Am Ende des Beobachtungsfensters sind kaum noch Container im Risiko, die Kurven sind dort grob. |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(C.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* je Container die beobachtete Zeit $t_i$ und die Ursache $c_i$ (0 zensiert, 1 LKW, 2 Bahn). *Gesucht:* der Anteil $F_1(t)$, der bis zur Zeit $t$ per LKW abgeholt wird.

| Symbol | Bedeutung |
|---|---|
| $t_i,\ c_i$ | Beobachtungszeit und Ursache des Containers $i$ (0 zensiert, 1 LKW, 2 Bahn) |
| $t_{(1)} < \dots < t_{(K)}$ | die verschiedenen Zeiten mit Abholung auf einem der beiden Wege |
| $n_k$ | Zahl der Container im Risiko zur Zeit $t_{(k)}$ (alle mit $t_i \ge t_{(k)}$) |
| $d_{1k},\ d_{2k}$ | Abholungen per LKW und per Bahn zur Zeit $t_{(k)}$ |
| $S(t)$ | Anteil, der zur Zeit $t$ noch auf keinem Weg abgeholt ist |
| $F_1(t),\ F_2(t)$ | kumulierte Inzidenz: Anteil, der bis $t$ per LKW bzw. per Bahn abgeholt wird |
| $h_1,\ h_2$ | konstante Abholraten (ursachenspezifische Raten) des LKW und der Bahn in der Simulation |
| $\hat{h}_1$ | geschätzte Rate des LKW unter konstanter Rate |

**Kumulierte Inzidenz nach Aalen–Johansen.** Der Anteil der noch Wartenden und die Beiträge einer Abholzeit:

$$\hat S(t) = \prod_{t_{(k)} \le t} \Big( 1 - \frac{d_{1k} + d_{2k}}{n_k} \Big)$$

$$\hat F_j(t) = \sum_{t_{(k)} \le t} \hat S(t_{(k-1)})\,\frac{d_{jk}}{n_k}, \qquad j = 1, 2$$

mit $\hat S(t_{(0)}) = 1$. Aus $\hat S(t_{(k)}) = \hat S(t_{(k-1)}) - \hat S(t_{(k-1)})\,(d_{1k} + d_{2k})/n_k$ folgt durch Aufsummieren

$$\hat F_1(t) + \hat F_2(t) + \hat S(t) = 1.$$

**Die naive Rechnung** (Bahn zensiert, 1 − Kaplan–Meier für den LKW):

$$1 - \hat S_{\text{naiv}}(t), \qquad \hat S_{\text{naiv}}(t) = \prod_{t_{(k)} \le t} \Big( 1 - \frac{d_{1k}}{n_k} \Big)$$

Weil $1 - (d_{1k} + d_{2k})/n_k \le 1 - d_{1k}/n_k$ in jedem Schritt gilt und $1 - \hat S_{\text{naiv}}(t) = \sum_{t_{(k)} \le t} \hat S_{\text{naiv}}(t_{(k-1)})\,d_{1k}/n_k$ ist, liegt $1 - \hat S_{\text{naiv}}(t)$ in jedem Datensatz
mindestens so hoch wie $\hat F_1(t)$; gleich sind beide, wenn die Bahn vor der letzten LKW-Abholung nichts abholt.

**Simulierte Welt und geschlossene Wahrheit.** Mit unabhängigen Zeiten $T_1 \sim \mathrm{Exp}(h_1)$ und $T_2 \sim \mathrm{Exp}(h_2)$ ist die Zeit bis zur ersten Abholung $\min(T_1, T_2) \sim \mathrm{Exp}(h_1 + h_2)$, also
$S(t) = e^{-(h_1 + h_2)\,t}$. Die Abholrate des LKW zur Zeit $u$ ist $h_1\,S(u)$, und

$$F_1(t) = \int_0^t h_1\, e^{-(h_1 + h_2)\,u}\,du = \frac{h_1}{h_1 + h_2}\,\Big( 1 - e^{-(h_1 + h_2)\,t} \Big).$$

Die naive Rechnung schätzt unter Unabhängigkeit der Zeiten $P(T_1 \le t) = 1 - e^{-h_1 t}$, den Anteil ohne Bahn. Die ursachenspezifische Rate des LKW (Exponential-Maximum-Likelihood, Zensierung eingeschlossen) ist

$$\hat h_1 = \frac{\sum_i \mathbf{1}[c_i = 1]}{\sum_i t_i},$$

und in dieser Welt geht $h_2$ nicht in sie ein, während $F_1(t)$ an beiden Raten hängt.

**Voraussetzungen und Konvention.** Zensierung unabhängig vom Ereignis; Risikomenge $\{ i : t_i \ge t_{(k)} \}$, bei gleicher Zeit zählt das Ereignis vor der Zensierung; gleichzeitige Abholungen auf beiden Wegen
zählen beide zur selben Zeit (die Tabelle in Experiment 1 zeigt ein Muster dazu).

**Literatur.** O. Aalen und S. Johansen (1978), An Empirical Transition Matrix for Non-Homogeneous Markov Chains Based on Censored Observations, Scandinavian Journal of Statistics 5, 141–150, und J. P. Fine und
R. J. Gray (1999), A Proportional Hazards Model for the Subdistribution of a Competing Risk, Journal of the American Statistical Association 94, 496–509: beide aus dem Gedächtnis genannt, die Originale sind
nicht eingesehen. A. A. Tsiatis (1975), A Nonidentifiability Aspect of the Problem of Competing Risks, Proceedings of the National Academy of Sciences 72(1), 20–22: im Original gelesen (Satz 2, Seite 21). Die Formeln
oben sind gegen unabhängige Rechnungen geprüft (`tests/test_oracle_konkurrierende_risiken.py`).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Überlebenszeit: zensierte Zeiten lesen](https://sebastianhanisch.net/konzepte-ueberlebenszeit.html)."
)
