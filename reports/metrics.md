# Raport dotyczący metryk dot. ewaluacji jakości klasyfikacji

## Kontekst biznesowy

Ważnym trade-offem jest FP vs FN. W przypadku wykrywania twóczości AI prawdopodobnie minimalizacja FP jest ważniejsza - może się to wiązać z fałszywym oskarżeniem.
Należy wziąć to pod uwagę podczas ewaluacji.

## Dotychczasowe metryki

- Accuracy
- Recall
- Precision
- F1 Score
- ROC AUC

### Dlaczego te metryki są wadliwe?

- Accuracy
  - fatalne dla niezbalansowanych klas
  - traktuje FP i FN tak samo
- F1
  - nie uwzględnia TN - może to maskować słabe wykrywanie klasy negatywnej
  - wrażliwość na zmianę etykiet - przy zmianie definicji klasy pozytywnej zmieni się wartość (przez brak uwzględnienia TN)
- ROC AUC
  - nie uwzględnia rozkładu danych
  - słabe dla niezbalansowanych klas - gdy dominuje klasa negatywna, duża liczba TN sztucznie utrzymuje FPR na niskim poziomie

## Przegląd zaawansowanych metryk

- Matthews Correlation Coefficient (MCC)
  - MCC = (TP × TN - FP × FN) / sqrt((TP + FP) × (TP + FN) × (TN + FP) × (TN + FN))
  - lepsze od F1 - wysoki wynik tylko, gdy klasyfikator dobrze przewiuje obydwie klasy
- Precision-Recall Curve / AUC
  - lepsze od ROC dla niezbalansowanych klas - Precision uwzględnia rozkład klas
  - pozwala na zoptymalizowanie progu odcięcia, żeby utrzymywać Precision na wysokim poziomie

- Brier Score
  - ocena kalibracji modelu
  - średnia kwadratów różnic między prawdopodobieństwem przewidywanym a faktycznym wynikiem
  - <0,1> - im niższy wynik, tym pewniejszy model
  - ocena czy surowe prawdopodobieńtwa wyliczane przez model dobrze odzwierciedlają rzeczywistość
