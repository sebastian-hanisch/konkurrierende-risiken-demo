"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_option_of_every_select_slider_and_selectbox_does_not_raise():
    """Auswahllisten: das Beobachtungsfenster (select_slider) und das Muster der sechs Container (selectbox) mit jedem erlaubten Wert."""
    for kind in ("select_slider", "selectbox"):
        count = len(getattr(_fresh_app(), kind))
        assert count, f"keine {kind} gefunden"
        for index in range(count):
            options = list(getattr(_fresh_app(), kind)[index].options)
            for option in options:
                at = _fresh_app()
                getattr(at, kind)[index].set_value(option).run(timeout=TIMEOUT)
                assert not at.exception, (kind, option, [str(e) for e in at.exception])


def test_app_zeigt_die_vier_experimente_den_annahmen_text_und_die_annahme_der_unabhaengigen_zensierung():
    at = _fresh_app()
    texts = " ".join([s.value for s in at.subheader] + [m.value for m in at.markdown])
    for marker in ("Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4", "Wo die Annahmen enden", "Die Zensierung ist unabhängig vom Ereignis"):
        assert marker in texts, marker
    labels = [e.label for e in at.expander]
    assert "📐 Mathematische Formulierung" in labels and "⚠️ Annahmen, Grenzen und bekannte Probleme" in labels
    assert at.metric and not at.error


def test_standardwelt_zeigt_die_wahrheit_und_beide_mittel_ohne_fehlermeldung():
    at = _fresh_app()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Wahrheit"] == "28,5 %"
    assert metrics["Kumulierte Inzidenz"] == "28,2 %" and metrics["1 − Kaplan–Meier"] == "38,9 %" and metrics["Zensiert"] == "26,0 %"
