# Konkurrierende Risiken: LKW oder Bahn – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-konkurrierende-risiken-demo.streamlit.app/)** -->

Teil der Linie **Überlebenszeit** (Zeit bis zum Ereignis bei Beobachtungen, die irgendwann abbrechen) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Der Aufhänger ist die Abholung von Containern: Ein Container wird auf **einem von zwei Wegen** abgeholt, per **LKW** oder per **Bahn**; wer zuerst kommt, nimmt dem anderen die Chance (**konkurrierende Risiken**).
Gefragt ist der **Anteil**, der bis Tag $t$ per LKW abgeholt wird. Kaplan–Meier kennt nur „Ereignis oder noch da“; die **kumulierte Inzidenz** (Aalen–Johansen) kennt zwei Ausgänge.
Die Demo führt das Verfahren an sechs Containern von Hand in Brüchen vor, hält es in einer simulierten Welt gegen die naive Rechnung (Bahn als zensiert, 1 − Kaplan–Meier) und gegen eine geschlossene Formel,
prüft, dass die Anteile zu 100 % zusammenpassen, und trennt die Frage nach dem **Anteil** von der Frage nach der **Rate**.

Gleicher Aufhänger Containerverweilzeit wie [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo), dort mit nur einem Ausgang und ohne Zensierung. Die Kante in der Linie: von Kaplan–Meier (Stück 1) zu zwei Ausgängen.
Die Raten sind ausgedachte Zahlen; es werden keine Zahlen aus echten Terminals behauptet.

**Annahme in dieser Demo, sichtbar in der App und hier: Die Zensierung ist unabhängig vom Ereignis.** Die simulierte Welt erfüllt sie; die Demo prüft sie nicht.

## Warum dieses Problem

Wer wissen will, wie viele Container bis Tag 5 per LKW abgeholt werden, hat Container, die schon per Bahn weg sind. Die naheliegende Rechnung behandelt die Bahn-Abholungen als **zensiert** und nimmt
1 − Kaplan–Meier für den LKW. In der Welt dieser Demo (LKW-Rate 0,10 je Tag, Bahn-Rate 0,15, Anteil bis Tag 5, 300 Container, Beobachtungsfenster 15 Tage, 26,0 % zensiert, 200 Läufe) liegt die
Wahrheit bei 28,5 %; die naive Rechnung kommt im Mittel auf 38,9 %, die kumulierte Inzidenz auf 28,2 %. Die naive Zahl liegt also um 10,4 Prozentpunkte über der Wahrheit
(36 % der Wahrheit), und in 199 von 200 Läufen liegt sie über der Wahrheit. Die Kernfrage ist: Wie viel Prozent der Container werden bis Tag $t$ per LKW abgeholt, und wie weit führt die
naive Rechnung in die Irre?

Gemessen ist außerdem, **wo der Fehler herkommt**: Die naive Rechnung bleibt im Mittel zwischen 38,6 % und 39,3 %, wenn die Bahn-Rate von 0 auf 0,40 steigt, während die
Wahrheit von 39,3 % auf 18,4 % fällt. Der Anteil, der per LKW abgeholt würde, **gäbe es die Bahn nicht**, ist 39,3 % (geschlossen 1 − e^(−0,10·5)). Die naive Rechnung liegt nahe bei diesem Anteil (um höchstens 0,8 Prozentpunkte darunter) und beantwortet
unter unabhängigen Abholzeiten diese andere Frage (Eigenmessung und eigene Lesart, nicht aus einer Quelle). Ohne Bahn (Rate 0, sonst Standardwelt) sind beide Rechnungen in allen 200 Läufen gleich.

## Befunde und Korrekturen gegenüber dem Plan

(Plan: `PLAN-08-konkurrierende-risiken.md` der Linien-Planung; Messung dort: 200 Läufe mit numpy-Zufall, Rohausgabe `out_uz_v1_bis_v8.txt`, Messung V4.)

1. **Zahlen weichen ab, es gilt die Zahl dieser Demo.** Vormessung: naive Rechnung 0,40 (+40 %), kumulierte Inzidenz 0,29, Wahrheit 0,285. Diese Demo (SplitMix64, Zufalls-Seed 35, 200 Läufe, gleiche Welt):
   naive Rechnung 38,9 % (36 % der Wahrheit mehr), kumulierte Inzidenz 28,2 %. Über sechs Zufalls-Seeds liegt das Mittel der naiven Rechnung zwischen 38,9 % und 39,5 %,
   das der kumulierten Inzidenz zwischen 28,2 % und 28,6 %. „+40 %“ steht nicht als Befund dieser Demo da; die Voraussetzung jeder Zahl ist LKW 0,10, Bahn 0,15, Tag 5, n = 300, Fenster 15 Tage (Entscheidung Z3).
2. **Der Plan sagt „zu hoch“, die Messung sagt genauer, wozu.** Die naive Zahl liegt für alle sechs Bahn-Raten der Reihe zwischen 38,6 % und 39,3 % (Tabelle unten), der Anteil „ohne Bahn“ ist 39,3 %. Das ist Eigenmessung in dieser Welt (unabhängige
   latente Abholzeiten, konstante Raten) und die Lesart dazu ist unser Schluss.
3. **Neu gegenüber dem Plan:** Zu jedem Datensatz gilt 1 − Kaplan–Meier (Bahn zensiert) ≥ kumulierte Inzidenz, mit Gleichheit genau dann, wenn die Bahn vor der letzten LKW-Abholung nichts abholt (Beweis im
   Expander der App; Eigenschaftstests an 60 Zufallsdatensätzen und an kleinen Datensätzen mit Gleichständen, 120 gezogen, davon die ohne LKW-Abholung ausgenommen). Der Plan nennt ihn nicht.
4. **Streuung ausgewiesen** (Plan: „Streuung der Schätzer nicht ausgewiesen“): Standardabweichung über die Läufe für n = 50, 100, 300, 600 (Tabelle unten).
5. **Experiment 4 anders gelesen als im Plan.** Der Plan fragt „Wie schnell geht er per LKW, falls er per LKW geht?“. Die ursachenspezifische Rate bezieht sich auf die **noch wartenden** Container, nicht auf
   die Teilmenge, die später per LKW abgeholt wird; die Demo fragt deshalb „Wie schnell holt der LKW die noch wartenden Container ab?“. Die Verteilung der Zeit unter der Bedingung „per LKW abgeholt“ wird nicht gezeigt.
6. **Fine–Gray nur als Satz** (Entscheidung Z2): kein Rechnen, keine Zahl.
7. **Orakel:** Der Plan nennt die geschlossene Formel und eine große Simulation. Hinzu kommt ein dritter Weg (Umverteilung nach rechts in exakten Brüchen, Abschnitt „Tests“), weil die Formel nur Erwartungswerte prüft
   und die Simulation nur mit Band.
8. **Kein Fremdcode-Orakel** (lifelines, statsmodels fehlen im Test-venv): Die kumulierte Inzidenz ist gegen die drei Rechenwege oben geprüft, nicht gegen eine Fremdbibliothek.

## Modell

Jeder Container hat zwei latente Zeiten bis zur Abholung: $T_1 \sim \mathrm{Exp}(h_1)$ für den LKW und $T_2 \sim \mathrm{Exp}(h_2)$ für die Bahn, unabhängig; abgeholt wird auf dem Weg mit der kleineren Zeit. Die Zensierzeit ist
zwischen 0 und dem Beobachtungsfenster gleichverteilt und von allem anderen unabhängig. Aus $S(t) = e^{-(h_1+h_2)t}$ folgt die **geschlossene Wahrheit**
$F_1(t) = \frac{h_1}{h_1+h_2}\,(1 - e^{-(h_1+h_2)t})$. Die Schätzer:

- **Kumulierte Inzidenz (Aalen–Johansen):** $\hat F_1(t) = \sum_{t_{(k)} \le t} \hat S(t_{(k-1)})\, d_{1k}/n_k$ mit dem Kaplan–Meier-Anteil $\hat S$ der noch Wartenden über beide Wege.
- **Naive Rechnung:** $1 - \prod (1 - d_{1k}/n_k)$, die Bahn-Abholungen als zensiert.

Formeln, Beweis der Ungleichung, Herleitung der geschlossenen Formel und Symboltabelle stehen im Expander „📐 Mathematische Formulierung“ der App, mit Lesehilfe. Gleichzeitige Abholungen zählen alle zur selben Zeit; bei
gleicher Zeit zählt das Ereignis vor der Zensierung. Der Kern (`kkr_core.py`) ist eine Kopie aus der Linien-Vorlage (`ueberlebenszeit-planung/kern/uz_core.py`, Stand 2026-11-10): SplitMix64-Zufall, Eingabeprüfung,
Kaplan–Meier und `cumulative_incidence`; `tests/test_core.py` vergleicht die Funktionstexte mit der Quelle, falls sie vorhanden ist.

## Methodik

Die App hat vier Experimente.

1. **Die Tabelle von Hand.** Sechs Container (Tag, Weg): (2, LKW), (3, Bahn), (4, LKW), (5, noch da), (6, Bahn), (8, LKW). Die App zeigt die Tabelle in exakten Brüchen (im Risiko, Abholungen je Weg, $S(t^-) \to S(t)$,
   kumulierte Inzidenz je Weg, naive Spalte), wächst Zeile für Zeile (Regler) und hat drei weitere Muster: ohne Konkurrenz (die Bahn holt nie ab), gleichzeitige Abholungen am Tag 4, Zensierung vor dem ersten Ereignis.
2. **Naiv gegen richtig** (Kernfrage). 200 Läufe mit festen Zufalls-Seeds; Regler in der Seitenleiste: Abholrate des LKW (0,02 bis 0,30 je Tag), Abholrate der Bahn (0 bis 0,40), Container n (50 bis 600),
   Beobachtungsfenster (5 bis 1000 Tage), Tag der Frage (1 bis 30), Zufalls-Seed. Gezeigt werden Wahrheit, beide Mittel mit Standardfehler, je Lauf ein Punkt im Diagramm, der Anteil „ohne Bahn“ und eine Reihe über
   sechs Bahn-Raten (0; 0,05; 0,10; 0,15; 0,25; 0,40; 200 Läufe je Rate, gleiche Zufallszahlen für LKW-Zeit und Zensierung).
3. **Die Summe stimmt.** Ein Datensatz (Lauf 1 der Welt): LKW, Bahn und „noch da“ als Fläche (zusammen 100 %) und die Summe der naiven Anteile beider Wege gegen die der kumulierten Inzidenz; dazu die Kurve des LKW,
   naiv gegen kumuliert gegen die Wahrheit.
4. **Zwei Fragen, zwei Antworten.** Aus denselben Daten: der Anteil per LKW (kumulierte Inzidenz, hängt an beiden Raten) und die ursachenspezifische Rate des LKW (Zahl der LKW-Abholungen geteilt durch die gesamte
   beobachtete Zeit aller Container, unter konstanter Rate; in dieser Welt geht die Bahn-Rate nicht in sie ein), gegen die Bahn-Rate aufgetragen.

Jede Meldung statt einer Zahl kommt aus dem Kern (`SurvivalError` mit verständlichem Text); die App fängt ihn ab und zeigt eine Fehlermeldung, nie einen Traceback und nie eine stille Zahl. Für die vielen Läufe
rechnet `estimate_fast` in `kkr_model.py` in einem sortierten Durchgang; `tests/test_model.py` hält es gegen die Kernfunktionen (auch mit Gleichständen), `tests/test_oracle_konkurrierende_risiken.py` gegen die
Umverteilung nach rechts.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen dieses Abschnitts stammen aus dem Code der Demo (SplitMix64, `kkr_model.py`) mit festen Zufalls-Seeds und stehen eingefroren in `tools/measure_konkurrierende_risiken_result.json`
(erzeugt mit `tools/measure_konkurrierende_risiken.py`). Welt, wenn nichts anderes steht: LKW-Rate 0,10, Bahn-Rate 0,15, Frage an Tag 5, n = 300 Container, Beobachtungsfenster 15 Tage, 200 Läufe; der
Zufalls-Seed der Läufe ist 35, Lauf r hat den Seed 35 + 1 000 003 · r. `tests/test_claims.py` rechnet alle Zellen mit dem Code der Demo neu, vergleicht sie mit der eingefrorenen Datei und prüft, dass die Tabellen
unten Wort für Wort aus ihr stammen (siehe „Tests“).

**Naiv gegen kumulierte Inzidenz (Messung 1, Standardwelt, 200 Läufe, 26,0 % zensiert).**

| Größe | Wert |
|---|---|
| Wahrheit (geschlossene Formel) | 28,5 % |
| Anteil „ohne Bahn“ (1 − e^(−0,10·5)) | 39,3 % |
| Kumulierte Inzidenz, Mittel ± Standardfehler des Mittels | 28,2 % ± 0,21 Prozentpunkte |
| 1 − Kaplan–Meier (Bahn zensiert), Mittel ± Standardfehler des Mittels | 38,9 % ± 0,29 Prozentpunkte |
| Standardabweichung über die Läufe: kumulierte Inzidenz / naive Rechnung | 2,93 Prozentpunkte / 4,15 Prozentpunkte |
| Läufe mit naiver Rechnung über der Wahrheit | 199 von 200 |
| Läufe mit kumulierter Inzidenz über der Wahrheit | 88 von 200 |
| Läufe mit naiver Rechnung unter der kumulierten Inzidenz | 0 von 200 |
| Größter Abstand naiv minus kumuliert in einem Lauf | 15,1 Prozentpunkte |

Die kumulierte Inzidenz liegt im Mittel 0,3 Prozentpunkte unter der Wahrheit (rund 1,5 Standardfehler); über sechs verschiedene Zufalls-Seeds (35, 101, 202, 303, 404, 505, je 200 Läufe) liegt ihr Mittel zwischen
28,2 % und 28,6 %, das der naiven Rechnung zwischen 38,9 % und 39,5 %. In den sechs Zufalls-Seeds liegt die naive Rechnung in 199 bis 200 von je 200 Läufen
über der Wahrheit, die kumulierte Inzidenz in 88 bis 102. Ob die kumulierte Inzidenz in dieser Welt eine kleine Verzerrung hat, ist mit diesen Zahlen nicht zu entscheiden; ein Befund dazu wird nicht behauptet.

**Reihe über die Bahn-Rate (Messung 2, 200 Läufe je Zeile, gleiche Zufallszahlen für LKW-Zeit und Zensierung).** Die naive Rechnung liegt in allen Zeilen zwischen 38,6 % und 39,3 %, die geschätzte Rate des LKW zwischen 0,098 und 0,100 (Wahrheit 0,10).

| Bahn-Rate | Wahrheit | Kumulierte Inzidenz (Mittel) | 1 − Kaplan–Meier (Mittel) | naive Rechnung über der Wahrheit | Rate des LKW, geschätzt (Mittel) |
|---|---|---|---|---|---|
| 0,00 | 39,3 % | 39,3 % | 39,3 % | 103 von 200 | 0,100 |
| 0,05 | 35,2 % | 35,0 % | 39,2 % | 177 von 200 | 0,100 |
| 0,10 | 31,6 % | 31,3 % | 39,1 % | 194 von 200 | 0,100 |
| 0,15 | 28,5 % | 28,2 % | 38,9 % | 199 von 200 | 0,099 |
| 0,25 | 23,6 % | 23,3 % | 38,9 % | 200 von 200 | 0,099 |
| 0,40 | 18,4 % | 18,0 % | 38,6 % | 199 von 200 | 0,098 |

Bei Bahn-Rate 0 sind beide Schätzer in allen 200 Läufen gleich (größter Unterschied unter 10⁻¹²). In der ersten Zeile liegen sie deshalb in denselben 103 von 200 Läufen über der Wahrheit; bei reinem Zufall
wären 100 ± 7 zu erwarten (Binomial-Standardabweichung bei 200 Läufen).

**Wie der Abstand mit dem Zeitpunkt der Frage wächst (Messung 3, Beobachtungsfenster 40 Tage, sonst Standardwelt).**

| Frage an Tag | Wahrheit | Kumulierte Inzidenz (Mittel) | 1 − Kaplan–Meier (Mittel) | Abstand der naiven Rechnung zur Wahrheit |
|---|---|---|---|---|
| 1 | 8,8 % | 8,8 % | 9,4 % | +0,6 Prozentpunkte |
| 2 | 15,7 % | 15,6 % | 18,0 % | +2,3 Prozentpunkte |
| 3 | 21,1 % | 20,9 % | 25,7 % | +4,6 Prozentpunkte |
| 5 | 28,5 % | 28,3 % | 39,1 % | +10,5 Prozentpunkte |
| 8 | 34,6 % | 34,4 % | 55,0 % | +20,4 Prozentpunkte |
| 10 | 36,7 % | 36,6 % | 63,0 % | +26,3 Prozentpunkte |

**Zensierung als Reihe (Messung 4, Frage an Tag 5, sonst Standardwelt, Beobachtungsfenster als Reihe).**

| Beobachtungsfenster (Tage) | zensiert (Mittel) | Wahrheit | Kumulierte Inzidenz (Mittel) | 1 − Kaplan–Meier (Mittel) |
|---|---|---|---|---|
| 5 | 57,1 % | 28,5 % | 28,4 % | 39,4 % |
| 10 | 36,8 % | 28,5 % | 28,1 % | 38,7 % |
| 15 | 26,0 % | 28,5 % | 28,2 % | 38,9 % |
| 25 | 15,8 % | 28,5 % | 28,3 % | 39,0 % |
| 40 | 9,9 % | 28,5 % | 28,3 % | 39,1 % |
| 1000 | 0,4 % | 28,5 % | 28,3 % | 39,1 % |

Die naive Rechnung liegt bei jedem Fenster zwischen 38,7 % und 39,4 %, auch bei 0,4 % zensierten Containern (Fenster 1000 Tage). Ohne Bahn dagegen ist sie sogar bei 51,7 % zensierten
Containern (Bahn-Rate 0, Fenster 15 Tage) gleich der kumulierten Inzidenz (Messung 2, erste Zeile). In dieser Welt liegt der Abstand also an der Konkurrenz und nicht an der Zensierung.

**Streuung nach Stichprobengröße (Messung 5, sonst Standardwelt).**

| Container n | Standardabweichung der kumulierten Inzidenz | Standardabweichung von 1 − Kaplan–Meier | naive Rechnung über der Wahrheit |
|---|---|---|---|
| 50 | 6,70 Prozentpunkte | 9,87 Prozentpunkte | 176 von 200 |
| 100 | 5,33 Prozentpunkte | 7,43 Prozentpunkte | 184 von 200 |
| 300 | 2,93 Prozentpunkte | 4,15 Prozentpunkte | 199 von 200 |
| 600 | 2,08 Prozentpunkte | 2,93 Prozentpunkte | 199 von 200 |

Bei jedem der vier $n$ ist die Standardabweichung der naiven Rechnung größer als die der kumulierten Inzidenz (gleiche Zufallszahlen in beiden Spalten).

**Die Summe der Anteile (Messung 6, Frage an Tag 5, Standardwelt).** In dem Datensatz, den Experiment 3 zeigt (Lauf 1, 300 Container, 19,3 % zensiert): naive Anteile LKW 46,2 % + Bahn 53,2 % +
noch da 25,2 % = **124,6 %**; kumulierte Inzidenz 33,4 % + 41,4 % + 25,2 % = 100,0 %. Über alle 200 Läufe liegt die naive Summe an Tag 5 in 200 von 200 Läufen
über 100 % (Mittel 120,4 %, Spanne 115,1 % bis 127,6 %); die Summe der kumulierten Inzidenz weicht in jedem Lauf höchstens um 2·10⁻¹⁵ von 1 ab.

**Presets (200 Läufe je Zeile).**

| Preset | Wahrheit | Kumulierte Inzidenz (Mittel) | 1 − Kaplan–Meier (Mittel) | naive Rechnung über der Wahrheit |
|---|---|---|---|---|
| Messung V4 (Standard) | 28,5 % | 28,2 % | 38,9 % | 199 von 200 |
| Ohne Konkurrenz (Bahn-Rate 0) | 39,3 % | 39,3 % | 39,3 % | 103 von 200 |
| Starke Konkurrenz (Bahn 0,30) | 21,6 % | 21,4 % | 38,9 % | 200 von 200 |
| Kaum zensiert (Fenster 1000 Tage) | 28,5 % | 28,3 % | 39,1 % | 198 von 200 |
| Später Zeitpunkt (Tag 10) | 36,7 % | 36,6 % | 63,0 % | 200 von 200 |
| Kleine Stichprobe (n = 50) | 28,5 % | 28,8 % | 39,1 % | 176 von 200 |

**Die Tabelle von Hand (Experiment 1, Handbeispiel; kein Zufall).** Dieselbe Tabelle zeigt die App in Brüchen; Prüfung gegen die Umverteilung nach rechts und gegen den Kern in `tests/`.

| Tag t | im Risiko | LKW | Bahn | S(t−) → S(t) | kumulierte Inzidenz LKW | kumulierte Inzidenz Bahn | naiv: 1 − Kaplan–Meier (LKW) |
|---|---|---|---|---|---|---|---|
| 2 | 6 | 1 | 0 | 1 → 5/6 | 1/6 | 0 | 1/6 |
| 3 | 5 | 0 | 1 | 5/6 → 2/3 | 1/6 | 1/6 | 1/6 |
| 4 | 4 | 1 | 0 | 2/3 → 1/2 | 1/3 | 1/6 | 3/8 |
| 6 | 2 | 0 | 1 | 1/2 → 1/4 | 1/3 | 5/12 | 3/8 |
| 8 | 1 | 1 | 0 | 1/4 → 0 | 7/12 | 5/12 | 1 |

Am Tag 8 steht die kumulierte Inzidenz des LKW bei 7/12 und die der Bahn bei 5/12 (zusammen 1), die naive Rechnung bei 1. Die weiteren Muster der App stehen mit allen Brüchen in `tests/test_model.py`.

## Ehrliche Grenzen

- **Eine Welt:** zwei Ursachen, konstante Raten, ein Beobachtungsfenster als Zensierung. Die Größen anderer Welten (steigende Raten, mehr Ursachen, abhängige Zeiten) sind nicht gemessen; nur die Rechenaussagen „naiv ≥ kumuliert“ und „Summe der Anteile = 1“
  gelten für jeden Datensatz.
- **„Ohne Bahn“ ist eine Lesart, die Unabhängigkeit braucht.** Ob die latenten Abholzeiten unabhängig sind, lässt sich aus (Zeit, Ursache) allein weder bestätigen noch widerlegen: Zu jeder glatten gemeinsamen
  Verteilung gibt es eine mit unabhängigen Zeiten, die dieselben beobachtbaren Teilüberlebensfunktionen liefert (Tsiatis 1975, PNAS 72(1), 20–22, Satz 2, Seite 21; Lesestufe Volltext (Scan der drei Seiten) laut dem Lektüre-Bericht der Linien-Planung). Die kumulierte Inzidenz braucht diese Unabhängigkeit nicht.
- **Keine Konfidenzbänder der kumulierten Inzidenz.** Die App zeigt die Streuung über Läufe, nicht die Unsicherheit eines einzelnen Datensatzes.
- **Am Ende des Beobachtungsfensters** sind kaum noch Container im Risiko; die Kurven werden dort grob. Die Frage an Tag 30 bei einem Fenster von 5 Tagen läuft ohne Fehler, die Kurve ist dann nach der letzten
  Abholung konstant fortgesetzt.
- **Die Rate in Experiment 4** setzt eine konstante Rate voraus (Exponential-Maximum-Likelihood); eine nichtparametrische ursachenspezifische Rate wird nicht gezeigt.
- **Fine–Gray** (Regression auf die Rate der Teilverteilung; Fine und Gray 1999, aus dem Gedächtnis genannt, nicht eingesehen) kommt nur als Satz vor und ist nicht gemessen.
- **Nur am Rechner geprüft** (Seitenaufruf im Browser, Regler über AppTest), nicht auf Mobilgeräten; Plotly-Diagramme nur angesehen, nicht mit Mausklicks durchgespielt.
- **Literatur:** Aalen und Johansen (1978) und Fine und Gray (1999) sind aus dem Gedächtnis genannt, die Originale sind nicht eingesehen; Tsiatis (1975) ist im Original gelesen.

## Annahmen, Grenzen und bekannte Probleme

(Derselbe Text steht in der App im Expander „⚠️ Annahmen, Grenzen und bekannte Probleme“; ein Test vergleicht beide.)

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

## Tests

Die Suite hat **142 Tests** (`pytest --collect-only -q tests`); Orakel-, Eigenschafts- und Modell-Tests laufen in wenigen Sekunden; den größeren Teil der Laufzeit bestimmen der AppTest und die Neuberechnung der Messzellen in `tests/test_claims.py`.

| Datei | Prüft |
|---|---|
| `tests/test_core.py` | kopierte Kernfunktionen (Kaplan–Meier, kumulierte Inzidenz, Eingabeprüfung, SplitMix64) gegen die Handbeispiele der Linie; Funktionstexte gegen die Quelle |
| `tests/test_model.py` | Handtabellen aller vier Muster in Brüchen, geschlossene Wahrheit, schnelle Schätzer gegen den Kern (150 Datensätze mit Gleichständen), Datenerzeugung, Läufe |
| `tests/test_oracle_konkurrierende_risiken.py` | drei unabhängige Wege: Umverteilung nach rechts in exakten Brüchen (200 + 200 + 200 Zufallsdatensätze mit Gleichständen, Handtabellen Bruch für Bruch), Quadratur der geschlossenen Formel (SciPy, übersprungen, wenn es fehlt), große Simulation (n = 40 000) mit Band |
| `tests/test_eigenschaften_konkurrierende_risiken.py` | Summe = 1, Monotonie, ohne Konkurrenz kumuliert = 1 − Kaplan–Meier, naiv ≥ kumuliert (und Gleichheit genau ohne Bahn vor der letzten LKW-Abholung), Umsortieren, Skalieren der Zeit, Vertauschen der Wege, gepaarte Welten |
| `tests/test_claims.py` | jede Zahl dieser README: Messzellen neu gerechnet gegen die eingefrorene Datei, Tabellen und Prosa-Zahlen gegen die README, Aussagen der Preset-Hilfen |
| `tests/test_verdicts.py` | jedes Verdikt der App an handgesetzten Werten (Schwellen, jede Zahl im Text) |
| `tests/test_edge_cases.py` | n = 1 und 2, gleichzeitige Abholungen, Zensierung vor dem ersten Ereignis, alle zensiert, nur eine Ursache, Frage hinter dem Fenster, Reglergrenzen, Meldungen |
| `tests/test_presets.py`, `test_layout.py` | Presets und Reglergrenzen; Schnellstart oben in der Seitenleiste, fester Kopf, Hervorhebung des aktiven Presets, jeder Regler hebt sie auf |
| `tests/test_app.py`, `test_footer.py`, `test_runtime_imports.py` | AppTest-Rauchtest (jeder Button, Regler an Minimum und Maximum, jede Auswahl), Footer, Laufzeit-Importe gegen `requirements.txt` |

**Zahl → Test.** Zahlen aus den Messungen 1 bis 6 und aus den Presets: eingefroren in der JSON-Datei und in `tests/test_claims.py` mit dem Code der Demo neu gerechnet (Band 1e-9, Zählungen exakt); die Tabellen der
README stammen Wort für Wort aus der Datei. Die Brüche der Handtabelle: exakt in `tests/test_model.py` und `tests/test_oracle_konkurrierende_risiken.py`. Die geschlossene Wahrheit (28,5 % und 39,3 % in
den Beispielen): `tests/test_model.py` (Handwert) und `tests/test_oracle_konkurrierende_risiken.py` (Quadratur, Simulation mit Band). Weitere Zahlen im Text sind Handwerte (Brüche), Rechnungen aus der geschlossenen Formel oder aus der JSON-Datei; die Plan-Zahlen aus der Vormessung (0,40; 0,29; +40 %) stehen nur als Vergleich im Abschnitt „Befunde und Korrekturen gegenüber dem Plan“.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: vier Experimente, Annahmen, Formeln |
| `kkr_core.py` | Kopie aus `ueberlebenszeit-planung/kern/uz_core.py` (SplitMix64, Eingabeprüfung, Kaplan–Meier, kumulierte Inzidenz); wird nicht verändert |
| `kkr_model.py` | geschlossene Wahrheit, Schätzer (kern- und sortiert), Handtabelle in Brüchen, Läufe, Reihen |
| `kkr_scenario.py` | Welt (World), Datenerzeugung mit SplitMix64, die vier Muster von Experiment 1 |
| `kkr_evaluation.py` | Formatierung, Zeitraster, Verdikte als reine Funktionen |
| `kkr_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets, Annahmen-Text |
| `kkr_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kkr_layout.py` | Fester Kopf (Stylesheet) |
| `kkr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/measure_konkurrierende_risiken.py` | Messskript; schreibt `tools/measure_konkurrierende_risiken_result.json` (eingefroren) und die Tabellen dieser README |
| `requirements-dev.txt` | Laufzeit plus pytest und SciPy (nur für die Quadratur im Orakel-Test) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): Streamlit, NumPy, Plotly |
| `tests/` | siehe „Tests“ |

## Bewusst nicht umgesetzt

- **Fine–Gray** und jede Regression mit Merkmalen der Container (kein Cox, keine Gruppenvergleiche); die Demo hat nur zwei Ursachen ohne Merkmale.
- **Konfidenzbänder** der kumulierten Inzidenz und Tests auf Gleichheit zweier kumulierter Inzidenzen.
- **Mehr als zwei Ursachen**, abhängige latente Zeiten und abhängige Zensierung (die Annahme der unabhängigen Zensierung wird in der Linie in einem eigenen Stück geprüft).
- **Nichtparametrische ursachenspezifische Raten** (Nelson–Aalen) und die bedingte Verteilung der Zeit unter „per LKW abgeholt“.
- **Echte Terminal-Daten:** Alle Raten sind ausgedachte Zahlen.
- **PDF-Export.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
