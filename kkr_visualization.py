"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kkr_constants as C
import kkr_model as M
from kkr_scenario import World


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.28), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


# ------------------------------------------------------------------------------------------------ Experiment 1


def build_hand(table: list[dict], selected: float | None):
    """Die Treppen der Handtabelle: kumulierte Inzidenz je Weg, noch da und die naive Kurve (LKW, Bahn als zensiert); senkrechte Linie bei der gewählten Zeit."""
    if not table:
        return _base(go.Figure())
    xs = [0.0] + [float(r["t"]) for r in table]

    def ys(key):
        return [0.0 if key != "s" else 1.0] + [float(r[key]) for r in table]

    x_end = xs[-1] + 1.0
    fig = go.Figure()
    for key, name, color, dash in (("cif1", "LKW: kumulierte Inzidenz", C.COLOR_LKW, "solid"), ("cif2", "Bahn: kumulierte Inzidenz", C.COLOR_BAHN, "solid"),
                                   ("s", "noch da", C.COLOR_REST, "solid"), ("naive1", "naiv: 1 − Kaplan–Meier (LKW)", C.COLOR_NAIVE, "dash")):
        fig.add_trace(go.Scatter(x=xs + [x_end], y=ys(key) + [ys(key)[-1]], mode="lines+markers", line=dict(color=color, width=3, shape="hv", dash=dash),
                                 marker=dict(size=7), name=name, hovertemplate="Tag %{x:g}: %{y:.4f}<extra>" + name + "</extra>"))
    if selected is not None:
        fig.add_vline(x=selected, line=dict(color=C.COLOR_LINE, dash="dot"), annotation_text=f"Tag {selected:g}", annotation_position="top")
    fig.update_xaxes(title_text="Tag", range=[0, x_end])
    fig.update_yaxes(title_text="Anteil der Container", range=[0, 1.05])
    return _base(fig)


# ------------------------------------------------------------------------------------------------ Experiment 2


def build_estimators(study: M.Study, s: M.Summary):
    """Je Lauf ein Punkt: die beiden Schätzungen zur Zeit t0 gegen die Wahrheit (durchgezogen) und gegen den Anteil ohne Bahn (gepunktet)."""
    fig = go.Figure()
    for values, name, color in ((study.cif_lkw, "Kumulierte Inzidenz (Aalen–Johansen)", C.COLOR_LKW), (study.naive_lkw, "1 − Kaplan–Meier, Bahn als zensiert", C.COLOR_NAIVE)):
        fig.add_trace(go.Box(y=np.round(values, 5), name=name, boxpoints="all", jitter=0.5, pointpos=0, marker=dict(color=color, size=4, opacity=0.6),
                             line=dict(color=color), fillcolor="rgba(0,0,0,0)", boxmean=True, hovertemplate="%{y:.4f}<extra>" + name + "</extra>"))
    # Die beiden Linien stehen in der Legende (ein Text im Diagramm läge über den Kästen).
    for y, name, dash in ((s.truth, "Wahrheit", "solid"), (s.truth_net, "Anteil ohne Bahn", "dot")):
        fig.add_hline(y=y, line=dict(color=C.COLOR_TRUTH, width=2, dash=dash))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash=dash), name=f"{name} {100 * y:.1f} %".replace(".", ","), hoverinfo="skip"))
    fig.update_yaxes(title_text=f"Anteil per LKW bis Tag {study.world.t0:g}", tickformat=".0%")
    return _base(fig, height=420)


def build_h2_sweep(rows: list[dict]):
    """Über die Bahn-Rate: Wahrheit, Mittel der kumulierten Inzidenz, Mittel der naiven Rechnung (jeweils Mittel über die Läufe) und der Anteil ohne Bahn."""
    h2 = [r["h2"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h2, y=[r["truth"] for r in rows], mode="lines", line=dict(color=C.COLOR_TRUTH, width=3), name="Wahrheit", hovertemplate="Bahn %{x:.2f}: %{y:.4f}<extra>Wahrheit</extra>"))
    fig.add_trace(go.Scatter(x=h2, y=[r["aj"] for r in rows], mode="markers", marker=dict(size=10, color=C.COLOR_LKW), name="Kumulierte Inzidenz (Mittel)",
                             hovertemplate="Bahn %{x:.2f}: %{y:.4f}<extra>Kumulierte Inzidenz</extra>"))
    fig.add_trace(go.Scatter(x=h2, y=[r["naive"] for r in rows], mode="markers", marker=dict(size=10, color=C.COLOR_NAIVE, symbol="diamond"), name="1 − Kaplan–Meier (Mittel)",
                             hovertemplate="Bahn %{x:.2f}: %{y:.4f}<extra>naiv</extra>"))
    fig.add_trace(go.Scatter(x=h2, y=[r["truth_net"] for r in rows], mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), name="Anteil ohne Bahn",
                             hovertemplate="%{y:.4f}<extra>ohne Bahn</extra>"))
    fig.update_xaxes(title_text="Abholrate der Bahn je Tag")
    fig.update_yaxes(title_text="Anteil per LKW", tickformat=".0%", rangemode="tozero")
    return _base(fig)


# ------------------------------------------------------------------------------------------------ Experiment 3


def build_curves(c: M.Curves, world: World):
    """Ein Datensatz über die Zeit: kumulierte Inzidenz und naive Kurve des LKW gegen die Wahrheit (durchgezogen) und gegen den Anteil ohne Bahn (gepunktet)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c.grid, y=M.truth_cif(world.h1, world.h2, c.grid), mode="lines", line=dict(color=C.COLOR_TRUTH, width=2), name="Wahrheit (LKW)",
                             hovertemplate="Tag %{x:.1f}: %{y:.4f}<extra>Wahrheit</extra>"))
    fig.add_trace(go.Scatter(x=c.grid, y=M.truth_net(world.h1, c.grid), mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), name="Anteil ohne Bahn",
                             hovertemplate="Tag %{x:.1f}: %{y:.4f}<extra>ohne Bahn</extra>"))
    fig.add_trace(go.Scatter(x=c.grid, y=c.cif_lkw, mode="lines", line=dict(color=C.COLOR_LKW, width=3, shape="hv"), name="LKW: kumulierte Inzidenz",
                             hovertemplate="Tag %{x:.1f}: %{y:.4f}<extra>kumulierte Inzidenz</extra>"))
    fig.add_trace(go.Scatter(x=c.grid, y=c.naive_lkw, mode="lines", line=dict(color=C.COLOR_NAIVE, width=3, dash="dash", shape="hv"), name="LKW naiv: 1 − Kaplan–Meier",
                             hovertemplate="Tag %{x:.1f}: %{y:.4f}<extra>naiv</extra>"))
    fig.add_vline(x=world.t0, line=dict(color=C.COLOR_LINE, dash="dot"), annotation_text=f"Tag {world.t0:g}", annotation_position="top")
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Anteil per LKW", tickformat=".0%", rangemode="tozero")
    return _base(fig)


def build_sum(c: M.Curves, world: World):
    """Links: die Anteile nach Aalen–Johansen als Fläche (LKW, Bahn, noch da): zusammen immer 1. Rechts: die Summe der naiven Anteile gegen die Summe der kumulierten Inzidenz."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Kumulierte Inzidenz: drei Anteile", "Summe der drei Anteile"), horizontal_spacing=0.1)
    for y, name, color in ((c.cif_lkw, "LKW", C.COLOR_LKW), (c.cif_bahn, "Bahn", C.COLOR_BAHN), (c.remaining, "noch da", C.COLOR_REST)):
        fig.add_trace(go.Scatter(x=c.grid, y=y, mode="lines", line=dict(width=0.5, shape="hv", color=color), stackgroup="eins", name=name,
                                 hovertemplate="Tag %{x:.1f}: %{y:.4f}<extra>" + name + "</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=c.grid, y=c.sum_naive, mode="lines", line=dict(color=C.COLOR_NAIVE, width=3, shape="hv"), name="Summe naiv (LKW + Bahn + noch da)",
                             hovertemplate="Tag %{x:.1f}: %{y:.4f}<extra>Summe naiv</extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=c.grid, y=c.sum_correct, mode="lines", line=dict(color=C.COLOR_LKW, width=3, dash="dash", shape="hv"), name="Summe kumulierte Inzidenz",
                             hovertemplate="Tag %{x:.1f}: %{y:.4f}<extra>Summe kumulierte Inzidenz</extra>"), row=1, col=2)
    fig.add_vline(x=world.t0, line=dict(color=C.COLOR_LINE, dash="dot"), row=1, col=1)
    fig.add_vline(x=world.t0, line=dict(color=C.COLOR_LINE, dash="dot"), row=1, col=2)
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Anteil der Container", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Summe der Anteile", tickformat=".0%", range=[0, max(1.3, float(np.max(c.sum_naive)) + 0.05)], row=1, col=2)
    return _base(fig, height=400)


# ------------------------------------------------------------------------------------------------ Experiment 4


def build_two_questions(rows: list[dict], t0: float):
    """Links: Anteil per LKW bis t0 über die Bahn-Rate (Wahrheit und Mittel der kumulierten Inzidenz). Rechts: die geschätzte Rate des LKW (Mittel ± Standardfehler) und die Wahrheit."""
    h2 = [r["h2"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Frage 1: Anteil bis Tag {t0:g}", "Frage 2: Rate je Tag"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=h2, y=[r["truth"] for r in rows], mode="lines", line=dict(color=C.COLOR_TRUTH, width=3), name="Wahrheit",
                             hovertemplate="Bahn %{x:.2f}: %{y:.4f}<extra>Wahrheit</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=h2, y=[r["aj"] for r in rows], mode="markers", marker=dict(size=10, color=C.COLOR_LKW), name="Kumulierte Inzidenz (Mittel)",
                             hovertemplate="Bahn %{x:.2f}: %{y:.4f}<extra>kumulierte Inzidenz</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=h2, y=[r["rate_lkw"] for r in rows], mode="markers", marker=dict(size=10, color=C.COLOR_LKW, symbol="diamond"), name="Rate des LKW, geschätzt (Mittel)",
                             error_y=dict(type="data", array=[r["rate_lkw_se"] for r in rows], visible=True), hovertemplate="Bahn %{x:.2f}: %{y:.4f}<extra>Rate</extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=h2, y=[r["rate_true"] for r in rows], mode="lines", line=dict(color=C.COLOR_TRUTH, width=3), name="Wahrheit (Rate)",
                             hovertemplate="%{y:.4f}<extra>Wahrheit</extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Abholrate der Bahn je Tag")
    fig.update_yaxes(title_text="Anteil per LKW", tickformat=".0%", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Rate des LKW je Tag", rangemode="tozero", row=1, col=2)
    return _base(fig, height=400)
