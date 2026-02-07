# Projekt Widzenia Komputerowego -- Klasyfikacja Sportów

## Cel projektu

Celem projektu było zaprojektowanie, wytrenowanie oraz porównanie
konwolucyjnych sieci neuronowych (CNN) w zadaniu wieloklasowej
klasyfikacji obrazów z wykorzystaniem danych rzeczywistych i
syntetycznych oraz różnych strategii augmentacji.

Projekt porównuje: - dane rzeczywiste vs syntetyczne, - różne warianty
augmentacji, - dwie architektury sieci CNN.

------------------------------------------------------------------------

## Klasy

Zadanie polega na klasyfikacji obrazów do jednej z 5 klas sportowych:

1.  football (piłka nożna)
2.  basketball (koszykówka)
3.  tennis
4.  boxing (boks)
5.  swimming (pływanie)

------------------------------------------------------------------------

## Struktura zbioru danych

Całkowita liczba obrazów: **220**

  Typ danych                          Liczba
  ----------------------------------- ---------
  Obrazy rzeczywiste (train + test)   110
  Obrazy syntetyczne (train)          110
  **Łącznie**                         **220**

### Stały zbiór testowy

-   20 obrazów rzeczywistych
-   4 obrazy na klasę
-   nigdy nie używany w treningu
-   identyczny dla wszystkich eksperymentów

### Zbiór treningowy

-   Real: 90 obrazów (18 na klasę)
-   Synthetic: 110 obrazów (22 na klasę)

------------------------------------------------------------------------

## Struktura katalogów

    data/
      train_real/
        football/
        basketball/
        tennis/
        boxing/
        swimming/
      train_synth/
        football/
        basketball/
        tennis/
        boxing/
        swimming/
      test_real/
        football/
        basketball/
        tennis/
        boxing/
        swimming/

------------------------------------------------------------------------

## Scenariusze eksperymentów (E1--E4)

  -----------------------------------------------------------------------
  Scenariusz                    Dane treningowe
  ----------------------------- -----------------------------------------
  E1                            Real + Synthetic

  E2                            Tylko Synthetic

  E3                            Tylko Real

  E4                            Wyrównane bez miksu (większy typ
                                przycięty do liczności mniejszego)
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Architektury CNN

W projekcie wykorzystano dwie różne architektury:

1.  **ResNet18**
2.  **MobileNetV2**

Oba modele były wstępnie wytrenowane (pretrained) i następnie
dostosowane do zadania 5‑klasowej klasyfikacji.

------------------------------------------------------------------------

## Warianty augmentacji (A0--A3)

  Wariant   Opis
  --------- -----------------------------
  A0        Brak augmentacji
  A1        Odbicie lustrzane (flip)
  A2        Zmiana jasności i kontrastu
  A3        Rotacja obrazu

------------------------------------------------------------------------

## Łączna liczba eksperymentów

Projekt obejmuje:

-   2 architektury
-   4 scenariusze danych
-   4 warianty augmentacji

Łącznie:

    2 × 4 × 4 = 32 eksperymenty

Wszystkie modele były oceniane na tym samym, stałym zbiorze testowym 20
obrazów rzeczywistych.

------------------------------------------------------------------------

## Metryki ewaluacji

Główna metryka: - Accuracy (dokładność klasyfikacji)

Dodatkowo: - macierz pomyłek - precision / recall / F1-score
(opcjonalnie)

------------------------------------------------------------------------

## Najważniejsze wyniki

Najlepsze rezultaty:

  Model         Scenariusz        Augmentacja   Accuracy
  ------------- ----------------- ------------- ----------
  MobileNetV2   E3 (tylko real)   A2            0.95
  MobileNetV2   E3 (tylko real)   A3            0.95

### Główne wnioski

1.  Dane rzeczywiste dały najlepsze wyniki.
2.  Augmentacja poprawiła generalizację modelu.
3.  MobileNetV2 osiągnął lepsze wyniki niż ResNet18 na małym zbiorze
    danych.

------------------------------------------------------------------------

## Jak uruchomić projekt

### 1. Instalacja zależności

    pip install torch torchvision pandas matplotlib scikit-learn pillow

### 2. Trening prostego modelu

    python src/train_simple.py

### 3. Uruchomienie pełnego zestawu eksperymentów

    python src/train_experiment.py

Wyniki zostaną zapisane w:

    results/results.csv
    results/results.xlsx

### 4. Generowanie wykresów

    python src/plot_by_model.py

Wykresy pojawią się w:

    results/

