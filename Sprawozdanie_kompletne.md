# Sprawozdanie z projektu — Wykrywanie naczyń krwionośnych dna siatkówki oka

**Skład grupy:** *[wpisz imiona]*
**Język programowania:** Python 3.12, środowisko Jupyter Notebook
**Baza danych:** CHASE_DB1
**Data:** 2026-05-17

---

## Spis treści

1. [Opis projektu i cel](#1-opis-projektu-i-cel)
2. [Baza danych](#2-baza-danych)
3. [Biblioteki i środowisko](#3-biblioteki-i-środowisko)
4. [Metoda 1 — Przetwarzanie obrazu (wymagania 3.0)](#4-metoda-1--przetwarzanie-obrazu-wymagania-30)
5. [Metoda 2 — Klasyfikator ML (wymagania 4.0)](#5-metoda-2--klasyfikator-ml-wymagania-40)
6. [Metoda 3 — Sieć neuronowa CNN (wymagania 5.0)](#6-metoda-3--sieć-neuronowa-cnn-wymagania-50)
7. [Wyniki i porównanie metod](#7-wyniki-i-porównanie-metod)
8. [Wnioski](#8-wnioski)

---

## 1. Opis projektu i cel

Celem projektu jest automatyczne wykrywanie naczyń krwionośnych na fotografiach dna siatkówki oka (ang. *retinal fundus images*). Dla każdego piksela obrazu algorytm musi odpowiedzieć na pytanie: **czy ten piksel należy do naczynia krwionośnego, czy do tła?** Jest to zadanie binarnej klasyfikacji pikselowej.

Badanie naczyń siatkówki ma duże znaczenie kliniczne — ich morfologia pozwala wykrywać i monitorować choroby takie jak cukrzyca, nadciśnienie tętnicze czy jaskra. Ręczne obrysowywanie naczyń przez lekarzy jest czasochłonne i subiektywne, stąd potrzeba automatyzacji.

W projekcie zrealizowano trzy podejścia o rosnącej złożoności:

| Poziom | Metoda | Cel |
|---|---|---|
| **3.0** | Przetwarzanie obrazu (filtr Frangiego) | Baseline — klasyczne metody wizyjne |
| **4.0** | Klasyfikator ML (Random Forest) | Nauka na cechach z wycinków obrazu |
| **5.0** | Głęboka sieć CNN (Keras) | Nauka reprezentacji bezpośrednio z pikseli |

Każda kolejna metoda jest porównywana z poprzednią na tym samym zbiorze testowym.

---

## 2. Baza danych

### 2.1 Charakterystyka

**CHASE_DB1** (Child Heart and Health Study in England) to publicznie dostępna baza fotografii dna oka dzieci. Obrazy pochodzą z badań przesiewowych przeprowadzonych w angielskich szkołach.

| Cecha | Wartość |
|---|---|
| Liczba obrazów | 28 (14 par lewe/prawe oko) |
| Rozdzielczość | 999 × 960 px |
| Format | JPEG (obrazy), PNG (maski) |
| Maski eksperckie | 2 na obraz (dwóch niezależnych annotatorów) |

### 2.2 Struktura plików

```
CHASE_DB1/
├── images/
│   ├── Image_01L.jpg   ← lewe oko pacjenta 01
│   ├── Image_01R.jpg   ← prawe oko pacjenta 01
│   └── ...             (łącznie 28 plików)
├── masks_1st/
│   ├── Image_01L_1stHO.png   ← maska eksperta 1 (używana jako ground truth)
│   └── ...
└── masks_2nd/
    ├── Image_01L_2ndHO.png   ← maska eksperta 2 (do porównania)
    └── ...
```

### 2.3 Podział train / test

W całym projekcie stosowany jest stały podział:

| Zbiór | Obrazy | Liczba |
|---|---|---|
| **Uczący (train)** | Image_01L/R … Image_10L/R | 20 obrazów |
| **Testowy (hold-out)** | Image_11L/R … Image_14L/R | 8 obrazów |

Obrazy testowe **nigdy** nie były używane podczas trenowania żadnej z metod.

### 2.4 Niezrównoważenie klas

Naczynia stanowią ok. **7% pikseli** wewnątrz pola widzenia (FOV) — klasy są mocno niezrównoważone. Naiwny model „wszystko = tło" osiągnie accuracy ≈ 93%, ale sensitivity = 0. Dlatego jako miarę główną stosujemy **G-mean = √(sensitivity × specificity)**, która karze ignorowanie mniejszościowej klasy.

---

## 3. Biblioteki i środowisko

### 3.1 Instalacja

```bash
pip install numpy pillow scikit-image scikit-learn pandas matplotlib tensorflow
```

### 3.2 Zestawienie bibliotek

| Biblioteka | Wersja | Do czego służy |
|---|---|---|
| `numpy` | ≥1.24 | Operacje macierzowe na pikselach |
| `Pillow` | ≥9.0 | Wczytywanie obrazów JPG/PNG |
| `scikit-image` | ≥0.21 | Filtr Frangiego, CLAHE, morfologia, hysteresis |
| `scikit-learn` | ≥1.3 | Random Forest, confusion matrix |
| `pandas` | ≥2.0 | Tabelaryczne zestawianie wyników |
| `matplotlib` | ≥3.7 | Wizualizacja wyników |
| `tensorflow` / `keras` | ≥2.13 | Budowa i trening sieci CNN |

---

## 4. Metoda 1 — Przetwarzanie obrazu (wymagania 3.0)

### 4.1 Idea ogólna

Metoda opiera się wyłącznie na klasycznych technikach przetwarzania obrazu — bez uczenia maszynowego. Pipeline przekształca obraz krok po kroku tak, by naczynia krwionośne stały się łatwo odróżnialne od tła, a następnie proguje wynik.

```
[Obraz RGB]
    ↓  kanał zielony + inwersja
    ↓  Gaussian blur
    ↓  CLAHE
    ↓  Filtr Frangiego (multi-scale)
    ↓  Hysteresis thresholding
    ↓  Morfologia + FOV
[Binarna maska naczyń]
```

### 4.2 Krok 1 — Wybór kanału zielonego i inwersja

Fotografia dna oka to obraz RGB. Każdy kanał wygląda inaczej:

- **Kanał R (czerwony):** Siatkówka jest naturalnie czerwona — kanał jest jasny, mało kontrastowy
- **Kanał B (niebieski):** Niebieskie światło słabo penetruje siatkówkę — ciemny, zaszumiony
- **Kanał G (zielony):** Hemoglobina silnie absorbuje światło zielone (~540 nm), więc naczynia są **wyraźnie ciemniejsze** niż otaczająca tkanka. Najlepszy kontrast.

Po wybraniu kanału G wykonujemy **inwersję** (`1.0 − G`): naczynia stają się jasne, tło ciemne. Filtr Frangiego szuka domyślnie jasnych struktur, więc ta transformacja jest konieczna.

```python
green = img_rgb[..., 1].astype(np.float32) / 255.0
inverted = 1.0 - green
```

### 4.3 Krok 2 — Rozmycie Gaussowskie (σ = 1.5)

Ten krok jest **kluczowy dla jakości wyników**. Bez niego filtr Frangiego reaguje na drobne struktury tła — teksturę naczyniówki widoczną między naczyniami — i generuje tysiące fałszywych detekcji.

Rozmycie gaussowskie z `σ=1.5` wygładza szum i mikroteksturę, ale nie niszczy struktur szerszych niż ~3 piksele (a najcieńsze naczynia mają 2–3 px szerokości).

**Wpływ na wyniki:** bez tego kroku specificity spada z ~0.95 do ~0.83.

```python
smoothed = gaussian(inverted, sigma=1.5)
```

### 4.4 Krok 3 — CLAHE (adaptacyjna equalizacja histogramu)

**Problem z klasyczną equalizacją histogramu:** działa globalnie — obszary jasne i ciemne rywalizują ze sobą, co prowadzi do utraty lokalnych szczegółów.

**CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) rozwiązuje ten problem:
1. Dzieli obraz na małe kafle (tiles)
2. Każdy kafel jest equalizowany niezależnie
3. Histogram ma narzucony limit (`clip_limit`) — zapobiega wzmacnianiu szumu w jednorodnych obszarach
4. Kafle są łączone z bilinearną interpolacją

Stosujemy **umiarkowany** `clip_limit=0.015`. Wyższa wartość nadmiernie wzmacniała teksturę tła, czyniąc je „naczyniopodobnym".

```python
clahe = exposure.equalize_adapthist(smoothed, clip_limit=0.015)
```

### 4.5 Krok 4 — Filtr Frangiego (multi-scale Hessian)

Filtr Frangiego (Frangi et al., 1998) wykrywa struktury **tubularne** — wyglądające jak rurki lub wałki.

**Matematyczna idea:**

Dla każdego piksela obliczana jest **macierz Hessego** (macierz drugich pochodnych intensywności):

$$H = \begin{pmatrix} I_{xx} & I_{xy} \\ I_{xy} & I_{yy} \end{pmatrix}$$

Wartości własne macierzy Hessego (λ₁, λ₂, gdzie |λ₁| ≤ |λ₂|) opisują lokalny kształt struktury:

| Wartości własne | Interpretacja |
|---|---|
| λ₁ ≈ 0, λ₂ ≈ 0 | Płaski obszar — brak struktury |
| λ₁ ≈ 0, λ₂ duże | Struktura **liniowa / rurkowata** ← naczynie! |
| λ₁ ≈ λ₂, oba duże | Struktura punktowa (blob) |

Filtr produkuje mapę „naczyniowości" (vesselness) w skali 0–1. Obliczenia powtarzane są dla wielu skali σ ∈ {1, 2, 3, 4, 5} pikseli, a wynik to maksimum po wszystkich skalach — dzięki temu wykrywane są zarówno cienkie, jak i grube naczynia.

```python
vesselness = frangi(prep, sigmas=range(1, 6), black_ridges=False)
```

### 4.6 Krok 5 — Maska pola widzenia (FOV)

Obraz dna oka jest okrągły — poza okiem mamy czarne piksele, które nie powinny być analizowane. Tworzymy maskę FOV:

```python
fov = np.mean(img_rgb, axis=2) > 20          # jasność > 20 = wewnątrz oka
fov = morphology.binary_erosion(fov, disk(5)) # zwężamy o 5px (krawędź FOV też wygląda jak linia)
```

### 4.7 Krok 6 — Hysteresis thresholding

Proste progowanie mapy vesselness ma wadę: wysoki próg gubi cienkie naczynia, niski próg przepuszcza za dużo szumu.

**Hysteresis thresholding** używa dwóch progów:
- **Wysoki (96. percentyl):** piksele pewne — „ziarna" naczyń
- **Niski (85. percentyl):** piksele kandydujące

Piksel kandydujący zostaje zaliczony do naczynia tylko wtedy, gdy jest **połączony** z jakimś ziarnem. Dzięki temu zachowujemy ciągłość naczyń bez wzmacniania izolowanego szumu.

Progi wyznaczane są adaptacyjnie z rozkładu vesselness wewnątrz FOV — algorytm działa poprawnie niezależnie od jasności konkretnego obrazu.

### 4.8 Krok 7 — Postprocessing morfologiczny

```python
binary = morphology.binary_closing(binary, disk(2))      # łączy przerwy w naczyniach
binary = morphology.remove_small_objects(binary, 100)    # usuwa artefakty < 100 px
binary = binary & fov_mask                               # zeruje piksele poza okiem
```

---

## 5. Metoda 2 — Klasyfikator ML (wymagania 4.0)

### 5.1 Idea ogólna

Zamiast ręcznie projektować reguły detekcji, uczymy klasyfikator na przykładach. Dla każdego piksela wyznaczamy wektor cech opisujący jego lokalny kontekst (wycinek 5×5 px), a klasyfikator decyduje: naczynie czy tło.

```
[Obraz RGB]
    ↓  preprocessing (jak w 3.0)
    ↓  filtr Frangiego (jak w 3.0)
    ↓  podział na wycinki 5×5 px
    ↓  ekstrakcja 13 cech z każdego wycinka
    ↓  Random Forest (100 drzew)
[Binarna maska naczyń]
```

### 5.2 Ekstrakcja cech z wycinków 5×5 px

Dla każdego piksela pobieramy wycinek 5×5 pikseli wyśrodkowany na tym pikselu. Etykietą wycinka jest wartość środkowego piksela maski eksperckiej (1 = naczynie, 0 = tło).

Z każdego wycinka obliczamy **13 cech**:

| # | Cecha | Opis |
|---|---|---|
| 1 | `mean` | Średnia intensywność pikseli wycinka |
| 2 | `std` | Odchylenie standardowe |
| 3 | `min` | Minimum intensywności |
| 4 | `max` | Maksimum intensywności |
| 5 | `range` | Rozstęp (max − min) |
| 6 | `mu20` | Moment centralny μ₂₀ — wariancja w osi x |
| 7 | `mu02` | Moment centralny μ₀₂ — wariancja w osi y |
| 8 | `mu11` | Moment centralny μ₁₁ — kowariancja |
| 9 | `mu30` | Moment centralny μ₃₀ — asymetria x |
| 10 | `mu03` | Moment centralny μ₀₃ — asymetria y |
| 11 | `H1` | Moment Hu H₁ = η₂₀ + η₀₂ (niezmiennik rotacji — „rozmiar" struktury) |
| 12 | `H2` | Moment Hu H₂ = (η₂₀ − η₀₂)² + 4η₁₁² (anizotropia — „wydłużenie") |
| 13 | `vesselness` | Wartość filtra Frangiego w danym pikselu |

**Momenty centralne** opisują kształt rozkładu intensywności wewnątrz wycinka — np. μ₂₀ duże oznacza, że intensywności są rozłożone wzdłuż osi x.

**Momenty Hu** są wyliczane ze znormalizowanych momentów centralnych: η_pq = μ_pq / (μ₀₀^(1+(p+q)/2)). Są niezmiennicze na skalowanie i rotację.

**Cecha `vesselness`** pochodzi bezpośrednio z filtra Frangiego — to gotowy, silny sygnał o wykrywalności naczynia w danym miejscu.

### 5.3 Undersampling — dlaczego i jak

Naczynia stanowią ~7% pikseli → bez balansowania model uczyłby się głównie rozpoznawać tło. Zastosowano **ręczny undersampling**:

- Dla każdego obrazu treningowego losujemy **1 000 pikseli naczyniowych** i **1 000 pikseli tła**
- Łączny zbiór uczący: 20 obrazów × 2 000 = **40 000 próbek**, 50% naczyń / 50% tła

### 5.4 Klasyfikator — Random Forest

**Dlaczego Random Forest, nie np. SVM czy kNN?**

| Kryterium | RF | SVM | kNN |
|---|---|---|---|
| Skalowanie cech | Nie wymaga | Wymaga | Wymaga |
| Czas treningu | Szybki (paralel.) | Wolny dla >10k | Brak (lazy) |
| Czas predykcji | Szybki | Szybki | Wolny dla >10k |
| Odporność na nadmiarowe cechy | Tak (feature importance) | Nie | Nie |
| Interpretowalność | Feature importances | Słaba | Brak |

Parametry: `n_estimators=100, max_depth=20, n_jobs=-1, random_state=42`

### 5.5 Feature importances (ważność cech)

Wynik z wytrenowanego modelu:

| Cecha | Ważność |
|---|---|
| `vesselness` | 0.199 |
| `max` | 0.163 |
| `mean` | 0.118 |
| `range` | 0.092 |
| `std` | 0.074 |
| `mu02` | 0.067 |
| ... | ... |
| `H2` | 0.002 |

**Interpretacja:** RF w największym stopniu opiera się na wyniku filtra Frangiego (gotowy sygnał o naczyniowości) oraz na statystykach jasności (max, mean). Momenty Hu (H1, H2) mają niską ważność — dla wycinków 5×5 px są mało informatywne.

---

## 6. Metoda 3 — Sieć neuronowa CNN (wymagania 5.0)

### 6.1 Idea ogólna

Sieć konwolucyjna uczy się sama wyodrębniać przydatne cechy z obrazu — nie trzeba ich ręcznie projektować jak w metodzie 4.0. CNN przetwarza wycinki **32×32 px** (większy kontekst niż 5×5 w RF) i produkuje dla każdego wycinka prawdopodobieństwo, że środkowy piksel jest naczyniem.

```
[Obraz RGB]
    ↓  preprocessing (jak w 3.0)
    ↓  podział na wycinki 32×32 px
    ↓  CNN (3 bloki konwolucyjne)
    ↓  Dense + sigmoid
[Mapa prawdopodobieństw] → progowanie 0.5 → [Binarna maska]
```

### 6.2 Architektura sieci

```
Input: (32, 32, 1)        ← wycinek 32×32 px, 1 kanał (szarość)
│
├─ Conv2D(32, 3×3, relu) + BatchNorm + MaxPool(2×2)    → (16, 16, 32)
├─ Conv2D(64, 3×3, relu) + BatchNorm + MaxPool(2×2)    → (8, 8, 64)
├─ Conv2D(128, 3×3, relu)                              → (8, 8, 128)
│
├─ GlobalAveragePooling2D                              → (128,)
├─ Dense(64, relu) + Dropout(0.4)                      → (64,)
└─ Dense(1, sigmoid)                                   → prawdopodobieństwo
```

Łącznie: ~200K parametrów.

**Dlaczego GlobalAveragePooling zamiast Flatten?**
Flatten(8×8×128) = 8192 → Dense(64) dodałby 500K parametrów i duże ryzyko przeuczenia. GAP uśrednia każdą mapę cech do jednej liczby — znacznie mniej parametrów, mniejsze przeuczenie.

**BatchNormalization** — normalizuje aktywacje po każdej warstwie konwolucyjnej. Przyspiesza uczenie i stabilizuje trening.

**Dropout(0.4)** — losowo wyłącza 40% neuronów warstwy Dense podczas treningu. Zapobiega przeuczeniu.

### 6.3 Dane treningowe dla CNN

- Wycinki **32×32 px** (większy kontekst przestrzenny niż w RF)
- **500 naczyń + 500 tła** na obraz treningowy (mniej niż w RF ze względu na większe wycinki)
- Łącznie: 20 × 1 000 = **20 000 wycinków**
- Ten sam podział train/test co w metodach 3.0 i 4.0

### 6.4 Trening

| Parametr | Wartość | Uzasadnienie |
|---|---|---|
| `optimizer` | Adam | Adaptacyjne tempo uczenia, dobry default |
| `loss` | binary_crossentropy | Standardowa strata dla binarnej klasyfikacji |
| `epochs` | 20 | Kompromis czas/jakość na CPU |
| `batch_size` | 64 | Stabilne gradienty, mieści się w RAM |
| `validation_split` | 0.1 | 10% zbioru uczącego jako walidacja |

### 6.5 Predykcja na pełnym obrazie

Podczas predykcji przetwarzamy **wszystkie piksele FOV** (ok. 700K na obraz). Dla każdego piksela wycinamy wycinek 32×32 i przekazujemy do sieci. Ze względu na ograniczenia pamięci RAM, przetwarzamy piksele **w partiach po 4096**:

```
dla każdej partii pikseli (4096):
    1. wytnij wycinki 32×32 px
    2. przekaż przez sieć → wektor 4096 prawdopodobieństw
    3. zapisz wyniki
→ próg 0.5 → binarna maska
```

Czas predykcji: ~30–60 sekund na obraz (CPU).

---

## 7. Wyniki i porównanie metod

### 7.1 Definicje miar

| Miara | Wzór | Co mierzy |
|---|---|---|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Ogólny % poprawnych klasyfikacji |
| **Sensitivity** | TP/(TP+FN) | Jaki % naczyń został wykryty |
| **Specificity** | TN/(TN+FP) | Jaki % tła został poprawnie odrzucony |
| **G-mean** | √(sens·spec) | Główna miara — odporna na niezrównoważenie klas |
| **Bal. Acc.** | (sens+spec)/2 | Alternatywa dla G-mean |

> **Uwaga:** Accuracy jest tutaj miarą drugorzędną. Klasy są niezrównoważone (~93% tło), więc naiwny model „wszystko = tło" osiągnąłby accuracy = 93% przy sensitivity = 0. G-mean i Balanced Accuracy karzą takie zachowanie.

### 7.2 Wyniki Baseline (Frangi) — zbiór testowy

| Obraz | Accuracy | Sensitivity | Specificity | G-mean | Bal. Acc. |
|---|---|---|---|---|---|
| Image_11L | 0.9072 | 0.6123 | 0.9321 | 0.7555 | 0.7722 |
| Image_11R | 0.9264 | 0.6935 | 0.9462 | 0.8100 | 0.8198 |
| Image_12L | 0.9098 | 0.5220 | 0.9570 | 0.7068 | 0.7395 |
| Image_12R | 0.9006 | 0.5255 | 0.9460 | 0.7051 | 0.7358 |
| Image_13L | 0.9079 | 0.5664 | 0.9419 | 0.7304 | 0.7542 |
| Image_13R | 0.9103 | 0.5434 | 0.9483 | 0.7178 | 0.7458 |
| Image_14L | 0.9296 | 0.6221 | 0.9642 | 0.7745 | 0.7932 |
| Image_14R | 0.9116 | 0.5187 | 0.9515 | 0.7025 | 0.7351 |
| **ŚREDNIA** | **0.9129** | **0.5755** | **0.9484** | **0.7378** | **0.7620** |

### 7.3 Wyniki Random Forest (4.0) — zbiór testowy hold-out

| Obraz | Accuracy | Sensitivity | Specificity | G-mean | Bal. Acc. |
|---|---|---|---|---|---|
| Image_11L | 0.8926 | 0.8653 | 0.8949 | 0.8800 | 0.8801 |
| Image_11R | 0.9044 | 0.8132 | 0.9121 | 0.8612 | 0.8626 |
| Image_12L | 0.8438 | 0.8460 | 0.8435 | 0.8447 | 0.8447 |
| Image_12R | 0.8414 | 0.9048 | 0.8337 | 0.8685 | 0.8693 |
| Image_13L | 0.9017 | 0.7625 | 0.9156 | 0.8356 | 0.8391 |
| Image_13R | 0.8743 | 0.7991 | 0.8820 | 0.8396 | 0.8406 |
| Image_14L | 0.8759 | 0.8722 | 0.8763 | 0.8743 | 0.8743 |
| Image_14R | 0.8141 | 0.8496 | 0.8104 | 0.8298 | 0.8300 |
| **ŚREDNIA** | **0.8685** | **0.8391** | **0.8711** | **0.8542** | **0.8551** |

### 7.4 Wyniki CNN (5.0) — zbiór testowy hold-out

> *Uzupełnij po uruchomieniu sekcji 5.0 w notebooku `projekt_kompletny.ipynb`.*

| Obraz | Accuracy | Sensitivity | Specificity | G-mean | Bal. Acc. |
|---|---|---|---|---|---|
| Image_11L | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Image_11R | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Image_12L | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Image_12R | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Image_13L | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Image_13R | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Image_14L | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Image_14R | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **ŚREDNIA** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

### 7.5 Porównanie wszystkich metod (średnia z 8 obrazów)

| Metoda | Accuracy | Sensitivity | Specificity | G-mean | Bal. Acc. |
|---|---|---|---|---|---|
| **Frangi (3.0)** | 0.9129 | 0.5755 | 0.9484 | 0.7378 | 0.7620 |
| **Random Forest (4.0)** | 0.8685 | 0.8391 | 0.8711 | 0.8542 | 0.8551 |
| **CNN (5.0)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Delta RF − Frangi | −0.044 | **+0.264** | −0.077 | **+0.116** | **+0.093** |
| Delta CNN − RF | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

### 7.6 Analiza porównawcza: Frangi vs Random Forest

**Frangi (3.0)**:
- Bardzo wysoka swoistość (0.95) — prawie nie myli tła z naczyniem
- Niska czułość (0.58) — gubi ok. 42% naczyń, szczególnie cienkie i obwodowe
- Główny problem: obszar tarczy nerwu wzrokowego i cienkie naczynia obwodowe

**Random Forest (4.0)**:
- Radykalnie lepsza czułość (+26 pp) — wykrywa zdecydowanie więcej naczyń
- Nieco niższa swoistość (−8 pp) — więcej fałszywych alarmów
- G-mean wzrósł o +11 pp — RF jest lepszy jako całość
- Powodem poprawy jest m.in. cecha `vesselness` (Frangi) jako wejście do RF — klasyfikator „kalibruje" wynik Frangiego w oparciu o kontekst lokalny

**Typowe błędy po RF:**
- Fałszywe pozytywy: tekstura wokół tarczy nerwu wzrokowego (większe niż w Frangim)
- Fałszywe negatywy: bardzo cienkie naczynia na obrzeżach siatkówki

---

## 8. Wnioski

### 8.1 Co zadziałało dobrze

1. **Rozmycie gaussowskie przed Frangim** — kluczowy krok, który eliminuje fałszywe detekcje z tekstury naczyniówki. Bez niego swoistość spada dramatycznie.
2. **CLAHE z umiarkowanym clip_limit** — wzmacnia kontrast lokalnie bez nadmiernego wzmacniania szumu.
3. **Hysteresis thresholding** — zachowuje ciągłość naczyń lepiej niż proste progowanie.
4. **Undersampling do 50/50** — kluczowy dla RF, sprawia że klasyfikator skupia się na rozróżnieniu naczyń od tła, a nie na klasie dominującej.
5. **Cecha vesselness jako wejście do RF** — reużycie obliczeń z metody 3.0 jako cecha dla 4.0, co znacznie poprawia wyniki.

### 8.2 Ograniczenia

- **Tarcza nerwu wzrokowego** — jasna okrągła struktura, której krawędzie wszystkie metody mylą z naczyniami
- **Cienkie naczynia obwodowe** — niski kontrast, słaby sygnał Frangiego, trudne nawet dla RF
- **Mały dataset** — 20 obrazów uczących to niewiele dla CNN; w praktyce siec neuronowe wymagają setek/tysięcy przykładów
- **Brak augmentacji** — obroty, odbicia lustrzane zwiększyłyby efektywnie rozmiar zbioru uczącego

### 8.3 Możliwe usprawnienia

| Kierunek | Opis |
|---|---|
| Więcej cech dla RF | Cechy kolorystyczne (kanał R, B), lokalne gradienty |
| Augmentacja danych | Obroty ±90°, odbicia, zmiany jasności dla CNN |
| Większe wycinki CNN | 48×48 lub 64×64 px — więcej kontekstu przestrzennego |
| Pełnoobrazowe UNet | Przetwarza cały obraz naraz, łapie globalny kontekst |
| Maskowanie tarczy nerwu | Wykryć i wykluczyć jasną tarczę z analizy |

---

## Literatura

- A. F. Frangi et al., *Multiscale vessel enhancement filtering*, MICCAI 1998
- P. Liskowski, K. Krawiec, *Segmenting Retinal Blood Vessels With Deep Neural Networks*, IEEE TMI 2016
- Baza CHASE_DB1: https://blogs.kingston.ac.uk/retinal/chasedb1/
- Dokumentacja scikit-image: https://scikit-image.org/
- Dokumentacja scikit-learn: https://scikit-learn.org/
- Dokumentacja TensorFlow/Keras: https://keras.io/
