# Metryki w klasyfikacji binarnej

Ten dokument opisuje najczęściej stosowane metryki w klasyfikacji binarnej, ich znaczenie, sposób obliczania oraz interpretację nietypowych wartości (np. duży `accuracy` przy złym `logloss`). Celem jest ułatwienie zrozumienia wyników modelu i ich potencjalnych implikacji (np. przetrenowanie, zła kalibracja, niezrównoważone dane).

---

##  Macierz pomyłek (Confusion Matrix)

Macierz pomyłek to podstawowy sposób wizualizacji działania modelu klasyfikacyjnego:

|                 | Predykcja: 0 (negatywna) | Predykcja: 1 (pozytywna) |
|-----------------|--------------------------|---------------------------|
| **Prawda: 0**   | TN – True Negative       | FP – False Positive       |
| **Prawda: 1**   | FN – False Negative      | TP – True Positive        |

Na podstawie tej macierzy obliczamy większość metryk klasyfikacyjnych.

---

##  Kluczowe metryki

### 1. **Accuracy (dokładność)** 
- **Opis**: Procent poprawnie sklasyfikowanych przykładów.
- **Wzór**:  
  $$
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  $$
- **Uwagi**: Może być mylące w przypadku niezrównoważonych danych, więc trzeba uważać z jego stosowaniem.

---

### 2. **Precision (precyzja)**
- **Opis**: Mierzy jak wiele pozytywnych predykcji jest poprawnych.
- **Wzór**:
  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$
- **Zastosowanie**: Minimalizacja fałszywych alarmów.

---

### 3. **Recall (czułość)**
- **Opis**: Jak wiele rzeczywistych pozytywnych przykładów zostało wykrytych?
- **Wzór:**$$
  \text{Recall} = \frac{TP}{TP + FN}
  $$
- **Zastosowanie**: Minimalizacja pominięć (FN).

---

### 4. **F1 Score**
- **Opis:** Metryka która mierzy dokładność modelu balansując precyzję i czułość.
- **Wzór:**$$
  F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
  $$
- **Zalety**: Równoważy precyzję i czułość.

---

### 5. **Log Loss (Logarithmic Loss)**
- **Opis:** Miara różnicy między przewidywanymi prawdopodbieństwami a rzeczywistymi wartościami przy klasyfikacji.
- **Wzór:**$$
  \text{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]
  $$
- **Zalety**: Uwzględnia skalibrowanie predykcji (pewność modelu).
- **Wysoki logloss + dobre accuracy**: wskazuje na **złą kalibrację** modelu.

---

### 6. **AUC - ROC**
- **Opis**: Prawdopodobieństwo, że model prawidłowo rozróżni klasę pozytywną od negatywnej. AUC to pole pod krzywą ROC.
- **Wzór:**$$
  \text{AUC} = \int_{0}^{1} TPR(FPR) \, dFPR
  $$
- **Interpretacja**:
  - 0.5 = klasyfikator losowy
  - 1.0 = idealny
- **Zalety**: Niezależna od progu, odporna na niezbalansowane dane.

---
### 7. **Brier Score**
- **Opis**: Mierzy średni kwadratowy błąd między przewidywanym prawdopodobieństwem a etykietą.
- **Wzór**:$$
  \text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2
  $$
- **Zakres**: od `0.0` (idealnie skalibrowany) do `1.0` (kompletnie błędny).
- **Zastosowanie**:
  - Ocena kalibracji modeli probabilistycznych.
  - Mniej agresywny niż log loss – nie karze tak bardzo za wysoką pewność błędnej predykcji.
- **Uwaga**: niski Brier Score sugeruje dobre **skalibrowanie** predykcji, nawet jeśli `accuracy` jest przeciętne.

---

##  Typowe przypadki i interpretacje

| Sytuacja                                   | Możliwa interpretacja                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------------------ |
| **Wysokie Accuracy + Wysoki LogLoss**      | Dobre klasy, zła kalibracja (niepewność predykcji).                                  |
| **Niska Precision, wysoka Recall**         | Dużo wykrytych przypadków, ale też wiele fałszywych alarmów.                         |
| **Wysoki F1, niski AUC**                   | Model działa dobrze przy danym progu, ale globalnie słabo rozróżnia klasy.           |
| **Niski LogLoss, niski Accuracy**          | Model pewny siebie, ale często się myli – może sugerować źle dobrane cechy lub próg. |
| **Różnice między trenowaniem a walidacją** | Możliwe **przetrenowanie** modelu.                                                   |

---
##  Przykład praktyczny

Załóżmy że model został przetestowany na 1000 przykładach:

|                 | Predykcja: 0 | Predykcja: 1 |
|-----------------|--------------|--------------|
| **Prawda: 0**   | 850 (TN)     | 50 (FP)      |
| **Prawda: 1**   | 40 (FN)      | 60 (TP)      |

#### Obliczone metryki:
- **Accuracy** = 91%  
- **Precision** ≈ 54.5%
- **Recall** = 60%
- **F1 Score** ≈ 0.57
- **LogLoss** – zakładamy wysoki z powodu niepewnych predykcji
- **AUC** – umiarkowany

**Wniosek**: Model dobrze przewiduje klasy, ale ma **złą kalibrację** – wskazane użycie wykresu kalibracji i ewentualna korekta (np. Platt scaling).

---

##  Kod: obliczanie metryk i wizualizacji (Python + scikit-learn)

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    calibration_curve,
)
import matplotlib.pyplot as plt

#  Przykładowe dane
y_true = np.array([0]*850 + [0]*50 + [1]*40 + [1]*60)  # 1000 przykładów
y_pred = np.array([0]*850 + [1]*50 + [0]*40 + [1]*60)  # binarne predykcje

# Probabilistyczne predykcje (symulacja niepewności)
y_prob = np.concatenate([
    np.random.uniform(0.1, 0.4, 850),  # TN
    np.random.uniform(0.6, 0.9, 50),   # FP
    np.random.uniform(0.1, 0.4, 40),   # FN
    np.random.uniform(0.6, 0.9, 60),   # TP
])

#  Obliczanie metryk
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("Log Loss:", log_loss(y_true, y_prob))
print("ROC AUC:", roc_auc_score(y_true, y_prob))

# Macierz pomyłek
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Krzywa kalibracji
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Calibration')
plt.title('Calibration Curve')
plt.xlabel('Średnie przewidywane prawdopodobieństwo')
plt.ylabel('Rzeczywisty odsetek klasy 1')
plt.legend()
plt.grid()
plt.show()
