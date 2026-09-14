# Modular Network Analyzer
Implementacja i badanie własnego generatora sieci syntetycznych ze strukturą społecznościową

## Zawartość repozytorium

- src: Katalog zawierający pełny kod źródłowy modelu i przeprowadzonych badań.
    - modele.py: Implementacja generatorów sieci - modelu własnego oraz modeli porównawczych (Erdős–Rényi, BA, Holme–Kim, SBM, LFR).
    - metryki.py: Metryki strukturalne (rozkład stopni, gronowanie, długość ścieżki, modularność, detekcja społeczności) oraz funkcje uśredniania i wizualizacji.
    - generuj_siec.py: Generowanie i wizualizacja pojedynczej instancji modelu własnego.
    - generuj_siec_fazy.py: Wizualizacja sieci po Fazie I i po Fazie II wzrostu.
    - exp_wplyw_beta.py: Wpływ parametru β na gronowanie, modularność i wykrywalność struktury.
    - exp_wplyw_gini.py: Wpływ nierówności rozmiarów grup na modularność i wykrywalność struktury.
    - exp_wplyw_m.py: Wpływ parametru m na wykładnik rozkładu, gronowanie, modularność i wykrywalność.
    - exp_stabilnosc_duze_N.py: Test stabilności metryk przy większym rozmiarze sieci (N=5000 vs 1000).
    - paszport_modelu.py: Zbiorcze zestawienie metryk modelu przy parametrach bazowych.
    - porownawcze_cddf.py: Analiza rozkładu potęgowego i porównanie CCDF z modelami klasycznymi.
    - sredni_stopien.py: Średni stopień węzła w funkcji kolejności dodania do sieci.
    - wykrywalnosc_vs_inne.py: Porównanie modularności, wykrywalności i metryk strukturalnych modeli.
