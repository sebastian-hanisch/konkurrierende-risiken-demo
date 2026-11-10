"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))   # measure_konkurrierende_risiken.py


@pytest.fixture
def six():
    """Sechs Container (Tag, Weg) mit Weg 0 = zensiert, 1 = LKW, 2 = Bahn. Von Hand gerechnet (siehe tests/test_model.py): am Tag 8 steht die kumulierte Inzidenz
    des LKW bei 7/12, die der Bahn bei 5/12, die naive Rechnung (Bahn zensiert) bei 1."""
    return ((2, 1), (3, 2), (4, 1), (5, 0), (6, 2), (8, 1))
