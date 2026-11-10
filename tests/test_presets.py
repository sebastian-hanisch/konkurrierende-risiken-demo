"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Aussagen der Hilfetexte gegen die Messung (tools/measure_konkurrierende_risiken_result.json)."""

import kkr_constants as C
import kkr_presets as P
from kkr_scenario import World


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_the_options():
    for name, p in C.PRESETS.items():
        assert C.H1_MIN <= p["h1"] <= C.H1_MAX and C.H2_MIN <= p["h2"] <= C.H2_MAX, name
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.T0_MIN <= p["t0"] <= C.T0_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["window"] in C.WINDOW_OPTIONS, name


def test_default_preset_equals_the_default_world():
    first = next(iter(C.PRESETS.values()))
    assert World(**first) == World()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_sweep_grid_and_series_are_inside_the_slider_bounds():
    assert all(C.H2_MIN <= h <= C.H2_MAX for h in C.SWEEP_H2) and C.SWEEP_H2[0] == 0.0   # die Reihe beginnt ohne Konkurrenz
    assert all(C.T0_MIN <= t <= C.T0_MAX for t in C.SWEEP_T0)


def test_presets_differ_from_each_other_in_at_least_one_setting():
    keys = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(keys)) == len(keys)
