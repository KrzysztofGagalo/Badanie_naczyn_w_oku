# Wykrywanie naczyń krwionośnych dna siatkówki oka

*Sprawozdanie z projektu (wymagania podstawowe — ocena 3.0) z elementami edukacyjnymi*

---

## 1. Skład grupy

- *[do uzupełnienia]*

---

## 2. Zastosowany język programowania i biblioteki

Projekt zrealizowano w języku **Python 3.12** w środowisku **Jupyter Notebook** (VS Code).

Wykorzystane biblioteki:

| Biblioteka | Zastosowanie |
|---|---|
| `numpy` | operacje na macierzach pikseli |
| `Pillow (PIL)` | wczytywanie obrazów JPG/PNG |
| `scikit-image` | filtry obrazu (Frangi, CLAHE, morfologia, progowanie) |
| `scikit-learn` | wyznaczanie macierzy pomyłek |
| `pandas` | tabelaryczne zestawianie wyników |
| `matplotlib` | wizualizacja |

Instalacja:

```bash
pip install numpy pillow scikit-image scikit-learn pandas matplotlib
```

---

## 3. Opis zastosowanych metod

### 3.1 Baza danych: CHASE_DB1

CHASE_DB1 (Child Heart and Health Study in England) to publiczna baza 28 obrazów dna oka pochodzących z badań dzieci. Każdy obraz ma rozdzielczość **999 × 960 pikseli** oraz dwie ręcznie wykonane maski eksperckie z zaznaczonymi naczyniami (1st observer, 2nd observer). W projekcie używamy maski 1. obserwatora jako *ground truth*.

Struktura danych w projekcie:

```
CHASE_DB1/
├── images/        # 28 plików .jpg (oryginalne fotografie)
├── masks_1st/     # 28 plików .png (maska eksperta 1 — ground truth)
└── masks_2nd/     # 20 plików .png (maska eksperta 2 — do porównania)
```

Naczynia stanowią średnio **~7% pikseli** wewnątrz pola widzenia, więc problem jest mocno **niezrównoważony** (tło dominuje liczebnie nad naczyniami).

### 3.2 Pipeline przetwarzania obrazu

Cały proces dzielimy na trzy etapy:

```
[Obraz RGB] → Preprocessing → Filtr Frangiego → Postprocessing → [Binarna maska naczyń]
```

### 3.3 Wstępne przetwarzanie (preprocessing)

W preprocessingu chodzi o to, by wzmocnić kontrast naczyń, **nie wzmacniając jednocześnie tekstury tła**. Siatkówka pod naczyniami ma własny mikroskopijny wzór (naczyniówka, *choroid*), który po zbyt mocnej equalizacji zaczyna wyglądać jak siatka naczyniopodobnych struktur — a filtr Frangiego się na to nabiera.

#### 3.3.1 Wybór kanału zielonego

Obraz RGB składa się z trzech kanałów: czerwonego (R), zielonego (G) i niebieskiego (B). Każdy z nich rejestruje światło o innej długości fali, a w fotografii dna oka kanały te wyglądają zaskakująco różnie:

- **Kanał R** — fotoreceptory rejestrujące czerwień są prześwietlone, bo dno oka jest naturalnie czerwonawo-pomarańczowe (krew, naczyniówka). Kanał R jest jasny i mało kontrastowy.
- **Kanał B** — niebieskie światło słabo penetruje siatkówkę, kanał jest ciemny i bardzo zaszumiony.
- **Kanał G** — najlepszy kompromis: hemoglobina silnie absorbuje światło zielone (~540 nm), więc naczynia są **wyraźnie ciemniejsze niż tło**. Dlatego kontrast naczynie-tło jest najwyższy właśnie w kanale zielonym.

```python
green = img_rgb[..., 1]   # wybór drugiego kanału (indeks 1)
```

#### 3.3.2 Inwersja

Filtr Frangiego, którego użyjemy do detekcji, domyślnie szuka **jasnych** struktur liniowych. Tymczasem naczynia w kanale G są ciemne. Najprościej zaradzić temu inwersją (zamianą jasności):

```python
inverted = 1.0 - green / 255.0    # naczynia stają się jasne
```

#### 3.3.3 Rozmycie gaussowskie

Ten krok jest **kluczowy dla jakości wyników**. Filtr Frangiego jest niezwykle czuły na drobne struktury — gdy puścimy go wprost na obraz po inwersji, zacznie reagować na piksel-poziomowy szum oraz na teksturę naczyniówki, którą widać między dużymi naczyniami. Efekt: tysiące drobnych fałszywych detekcji.

Rozwiązanie: przed Frangim wygładzamy obraz **filtrem gaussowskim** z `sigma=1.5`. To rozmazuje szum oraz drobną teksturę tła, ale nie niszczy struktur o szerokości większej niż ~3 piksele (a takie są nawet najcieńsze naczynia w CHASE).

```python
from skimage.filters import gaussian
smoothed = gaussian(inverted, sigma=1.5)
```

Bez tego kroku specificity spada z ~0.95 do ~0.83 (czyli 12% pikseli tła to fałszywe alarmy) — to dramatyczna różnica.

#### 3.3.4 CLAHE — adaptacyjna equalizacja histogramu

**Equalizacja histogramu** to klasyczna technika rozciągania kontrastu — przekształcamy obraz tak, aby rozkład jasności był równomierny. Niestety klasyczna wersja działa globalnie: gdy część obrazu jest jasna a część ciemna, jasna część "zjada" ciemną i traci się szczegóły lokalne.

**CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) działa inaczej:

1. Dzieli obraz na małe kafle (tiles)
2. Każdy kafel ma swój własny histogram, który jest equalizowany niezależnie
3. Histogram ma narzucony limit (*clip limit*) — bardzo wysokie słupki są obcinane, a ich nadmiar redystrybuowany. Dzięki temu szum w jednolitych obszarach nie jest wzmacniany do absurdu
4. Kafle są łączone z bilinearną interpolacją, żeby uniknąć widocznych krawędzi

Efekt: kontrast lokalny jest mocno wzmocniony, ale szum pozostaje w ryzach.

W naszej implementacji stosujemy **umiarkowane** wzmocnienie kontrastu (`clip_limit=0.015`). Pierwsza wersja używała `clip_limit=0.03`, ale to zbyt mocno wzmacniało teksturę tła — wszystko, co minimalnie ciemniejsze, stawało się "naczyniopodobne". Mniejszy `clip_limit` daje słabsze wzmocnienie ale zachowuje czystość obrazu.

```python
from skimage import exposure
clahe = exposure.equalize_adapthist(smoothed, clip_limit=0.015)
```

#### 3.3.5 Maska pola widzenia (FOV)

Obraz dna oka to okrągły obraz oka osadzony w prostokątnej fotografii — wokół oka są czarne piksele (poza polem widzenia urządzenia). Te piksele nie należą do siatkówki i nie powinny brać udziału w analizie. Tworzymy więc **maskę FOV**:

```python
gray = np.mean(img_rgb, axis=2)              # uśredniona jasność
fov = gray > 20                              # piksele jaśniejsze niż próg = wnętrze oka
fov = morphology.binary_erosion(fov, morphology.disk(5))  # zwężamy o 5px,
                                              # by usunąć krawędź FOV (która zachowuje
                                              # się jak liniowa struktura i myli Frangiego)
```

### 3.4 Filtr Frangiego — wykrywanie struktur naczyniowych

Filtr Frangiego (Frangi et al., 1998) to metoda detekcji **struktur tubularnych** (czyli wyglądających jak rurki / wałki). Idea matematyczna:

Dla każdego piksela obrazu liczymy lokalnie **macierz Hessego** (macierz drugich pochodnych):

$$
H = \begin{pmatrix} I_{xx} & I_{xy} \\ I_{xy} & I_{yy} \end{pmatrix}
$$

Gdzie $I_{xx}$, $I_{yy}$, $I_{xy}$ to drugie pochodne intensywności obrazu w danym punkcie. Pochodne te liczymy po wcześniejszym wygładzeniu obrazu gaussianem o pewnej skali σ — to definiuje "rozmiar" struktur, których szukamy.

Macierz Hessego ma **wartości własne** λ₁, λ₂ (uporządkowane: |λ₁| ≤ |λ₂|). Ich relacja mówi nam, jak wygląda lokalnie obraz:

- |λ₁| ≈ |λ₂| ≈ 0 → płaski obszar (brak struktury)
- |λ₁| ≈ 0, |λ₂| duże → struktura **liniowa / tubularna** (naczynie!)
- |λ₁| ≈ |λ₂|, oba duże → struktura punktowa/blob

Frangi definiuje funkcję "naczyniowości" (vesselness), która jest duża właśnie w drugim przypadku. Wartość ta jest liczona dla wielu skal σ (różne grubości naczyń) i bierzemy maksimum:

$$
V(x) = \max_{\sigma \in [\sigma_{min}, \sigma_{max}]} V_\sigma(x)
$$

W skrócie: piksel jest "bardzo naczyniowy", jeśli na jakiejś skali wygląda jak część rurki.

```python
from skimage.filters import frangi
vesselness = frangi(prep, sigmas=range(1, 6), black_ridges=False)
```

Parametr `sigmas=range(1, 6)` oznacza skale σ ∈ {1, 2, 3, 4, 5} pikseli — odpowiada to różnym grubościom naczyń (od cienkich do grubych). `black_ridges=False` mówi, że szukamy *jasnych* struktur (po inwersji).

Wynik filtra to mapa wartości naczyniowości w przedziale [0, 1]. Aby otrzymać binarną maskę, musimy ją zaprogowć.

### 3.5 Postprocessing — od mapy naczyniowości do binarnej maski

#### 3.5.1 Hysteresis thresholding (progowanie z histerezą)

Najprostsze progowanie to: "vesselness > t → naczynie". Problem: jeśli próg jest wysoki, gubimy cienkie naczynia (niska czułość); jeśli niski — wykrywamy mnóstwo szumu (niska swoistość).

**Hysteresis thresholding** używa **dwóch progów**:

1. Wysokiego (`high`) — wszystkie piksele powyżej niego to "ziarna" pewnych naczyń
2. Niskiego (`low`) — wszystkie piksele powyżej niego są kandydatami

Następnie z ziaren "wyrastają" naczynia: kandydat jest zaliczany do naczyń tylko wtedy, gdy jest **połączony** (sąsiaduje przez piksele kandydatów) z jakimś ziarnem. Dzięki temu zachowujemy ciągłość naczyń (cienki fragment połączony z grubym jest zaliczony), ale odrzucamy izolowany szum (pojedyncze jasne piksele bez kontekstu).

Progi wyznaczamy adaptacyjnie na podstawie percentyli — to czyni algorytm odpornym na różnice jasności między obrazami:

Progi wybieramy **wysokie** — 85 percentyl (low) i 96 percentyl (high). Niższe progi (np. 75/92) powodują, że zbyt wiele "ledwo widocznych" pikseli przechodzi przez filtr i bywa zaliczonych do naczyń, drastycznie zwiększając liczbę fałszywych pozytywów.

```python
from skimage.filters import apply_hysteresis_threshold
low  = np.percentile(vesselness[fov], 85)   # 85 percentyl
high = np.percentile(vesselness[fov], 96)   # 96 percentyl
binary = apply_hysteresis_threshold(vesselness, low, high)
```

#### 3.5.2 Morfologia matematyczna

Po progowaniu maska bywa "dziurawa". Stosujemy operacje **morfologii matematycznej**:

- **Erozja** — usuwa cienkie elementy
- **Dylatacja** — powiększa elementy
- **Otwarcie** (erozja → dylatacja) — usuwa małe artefakty, zachowując kształt dużych obiektów
- **Zamknięcie** (dylatacja → erozja) — wypełnia małe dziury i łączy bliskie fragmenty

W naszym pipeline'ie stosujemy **zamknięcie** z dyskiem o promieniu 2, aby połączyć drobne przerwy w naczyniach:

```python
binary = morphology.binary_closing(binary, morphology.disk(2))
```

#### 3.5.3 Usunięcie małych obiektów

Każdy spójny obszar mniejszy niż 100 pikseli traktujemy jako artefakt i usuwamy. Wartość ta została dobrana empirycznie — większe `min_size` redukuje fałszywe pozytywy, ale przy zbyt dużym progu zaczynamy gubić krótkie boczne odgałęzienia naczyń.

```python
binary = morphology.remove_small_objects(binary, min_size=100)
```

#### 3.5.4 Zastosowanie maski FOV

Na koniec zerujemy wszystko poza okręgiem oka:

```python
binary = binary & fov_mask
```

### 3.6 Pełna funkcja segmentacji

Cały pipeline można zapisać jako jedną funkcję:

```python
def segment_vessels(img_rgb):
    fov  = get_fov_mask(img_rgb)
    prep = preprocess(img_rgb)                # G → inwersja → blur → CLAHE
    vess = frangi(prep, sigmas=range(1,6),
                  black_ridges=False)         # mapa naczyniowości
    return postprocess(vess, fov), fov        # hysteresis + morfologia + FOV
```

---

## 4. Wizualizacja wyników

Dla każdego analizowanego obrazu prezentujemy cztery widoki obok siebie:

1. **Oryginał** — kolorowa fotografia dna oka
2. **Ground truth** — maska ekspercka (czarno-biała, naczynia = białe)
3. **Predykcja** — wynik naszego algorytmu (czarno-biała maska)
4. **Nakładka** — fotografia z nałożoną kolorową analizą błędów:
   - 🟢 **zielony** = TP (true positive) — poprawnie wykryte naczynie
   - 🔴 **czerwony** = FP (false positive) — błędnie zaklasyfikowane jako naczynie (tło → naczynie)
   - 🔵 **niebieski** = FN (false negative) — przegapione naczynie (naczynie → tło)

```python
def make_overlay(rgb, pred, gt, fov_mask):
    overlay = rgb.copy()
    pred_m, gt_m = pred & fov_mask, gt & fov_mask
    overlay[pred_m & gt_m]   = [0, 255, 0]     # TP — zielony
    overlay[pred_m & ~gt_m]  = [255, 0, 0]     # FP — czerwony
    overlay[~pred_m & gt_m]  = [0, 0, 255]     # FN — niebieski
    return overlay
```

Wizualizacja pozwala szybko zlokalizować typy błędów: nadmiar czerwieni → algorytm "widzi naczynia tam, gdzie ich nie ma" (np. krawędzie tarczy nerwu wzrokowego, plamki); nadmiar niebieskiego → gubi cienkie naczynia.

W notebooku zaprezentowano wizualizacje 5+ obrazów testowych (Image_11L do Image_14R) — pierwsze pięć pokazuje przykłady względnie udanej segmentacji oraz typowe porażki.

---

## 5. Analiza wyników — metryki i interpretacja

### 5.1 Definicje miar

Dla problemu binarnego (naczynie = klasa pozytywna, tło = klasa negatywna) confusion matrix wygląda tak:

| | predykcja: naczynie | predykcja: tło |
|---|---|---|
| **ground truth: naczynie** | TP | FN |
| **ground truth: tło** | FP | TN |

Wyznaczane miary:

| Miara | Wzór | Interpretacja |
|---|---|---|
| Accuracy (trafność) | (TP+TN)/(TP+TN+FP+FN) | % poprawnie sklasyfikowanych pikseli |
| Sensitivity (czułość, TPR, recall) | TP/(TP+FN) | jaki % naczyń algorytm wykrył |
| Specificity (swoistość, TNR) | TN/(TN+FP) | jaki % tła algorytm poprawnie odrzucił |
| G-mean | √(sens · spec) | średnia geometryczna — wrażliwa na niezrównoważenie klas |
| Balanced accuracy | (sens + spec) / 2 | średnia arytmetyczna — alternatywa dla g-mean |

**Dlaczego dwie miary "zrównoważone"?** Klasy są mocno niezrównoważone (~93% tło, ~7% naczynia). Naiwny klasyfikator "wszystko to tło" osiągnąłby accuracy = 93%, co wygląda świetnie, a jest bezużyteczne. Miary g-mean i balanced accuracy karzą takie zachowanie, bo czułość tego klasyfikatora wynosi 0.

### 5.2 Wyniki na zestawie testowym (Image_11L — Image_14R, 8 obrazów)

| Obraz | Accuracy | Sensitivity | Specificity | G-mean | Bal. Acc. |
|---|---|---|---|---|---|
| Image_11L | 0.9070 | 0.6123 | 0.9320 | 0.7554 | 0.7721 |
| Image_11R | 0.9264 | 0.6935 | 0.9462 | 0.8100 | 0.8198 |
| Image_12L | 0.9097 | 0.5238 | 0.9567 | 0.7079 | 0.7402 |
| Image_12R | 0.9004 | 0.5255 | 0.9458 | 0.7050 | 0.7357 |
| Image_13L | 0.9079 | 0.5664 | 0.9419 | 0.7304 | 0.7542 |
| Image_13R | 0.9103 | 0.5434 | 0.9483 | 0.7178 | 0.7458 |
| Image_14L | 0.9298 | 0.6242 | 0.9641 | 0.7757 | 0.7941 |
| Image_14R | 0.9114 | 0.5187 | 0.9513 | 0.7025 | 0.7350 |
| **ŚREDNIA** | **0.9129** | **0.5760** | **0.9483** | **0.7381** | **0.7621** |

### 5.2a Porównanie z pierwszą wersją algorytmu

Pierwsza wersja pipeline'u (bez rozmycia gaussowskiego, `clip_limit=0.03`, progi p75/p92, `min_size=30`) dawała znacznie gorsze wyniki — generowała mnóstwo fałszywych pozytywów w teksturze naczyniówki:

| Wersja | Acc | Sens | Spec | G-mean |
|---|---|---|---|---|
| v1 (bez rozmycia, niskie progi) | 0.80 | 0.45 | 0.84 | 0.62 |
| **v2 (obecna, z rozmyciem)** | **0.91** | **0.58** | **0.95** | **0.74** |

Główne zmiany, które dały największą poprawę:
1. **Rozmycie Gaussa (σ=1.5) przed filtrem Frangiego** — usuwa drobny szum i teksturę naczyniówki
2. **Mniej agresywny CLAHE** (`clip_limit` 0.03 → 0.015) — nie wzmacnia tła do tego stopnia, że wygląda jak naczynia
3. **Wyższe progi hysteresis** (p75/p92 → p85/p96) — odrzuca słabe sygnały z obszarów tła
4. **Większe `min_size`** (30 → 100) — eliminuje krótkie fragmenty fałszywych pozytywów

### 5.3 Interpretacja

- **Accuracy ~91%** — wysoki wynik, ale pamiętajmy o niezrównoważeniu klas (naiwny model "wszystko = tło" miałby już ~93%).
- **Specificity ~95%** — algorytm bardzo dobrze odrzuca tło. Tylko ~5% pikseli tła generuje fałszywy alarm.
- **Sensitivity ~58%** — wykrywamy ok. 58% naczyń. Główne grube naczynia są łapane prawie w całości, natomiast cienkie naczynia obwodowe są często gubione (kontrast za mały, sygnał Frangiego nie przekracza progu).
- **G-mean ~0.74, Balanced acc ~0.76** — solidny wynik dla podejścia opartego wyłącznie na przetwarzaniu obrazu (literatura podaje wartości 0.60–0.80). Sieci neuronowe (UNet) osiągają tu g-mean ~0.85–0.90.

### 5.4 Typowe błędy algorytmu

**Fałszywe pozytywy (czerwony na nakładce):**

- **Tarcza nerwu wzrokowego** — duża jasna plama, której krawędzie filtr Frangiego błędnie identyfikuje jako naczynie
- **Krawędź FOV** — pomimo erozji maski, czasem pozostaje cieniutka linia, która jest tubularna
- **Hiperpigmentacje, refleksy świetlne** — lokalne struktury liniowe niebędące naczyniami

**Fałszywe negatywy (niebieski na nakładce):**

- **Cienkie naczynia obwodowe** — najczęstsze źródło błędu; sygnał Frangiego dla σ=1 jest słaby
- **Naczynia w obszarach o niskim kontraście** — szczególnie w plamie żółtej (centralna ciemna część siatkówki)
- **Rozwidlenia naczyń** — w punktach bifurkacji geometria nie jest już idealnie tubularna

### 5.5 Możliwe usprawnienia

1. **Strojenie parametrów** — zakres `sigmas`, progi percentyli, `min_size`. Można dobierać siatką (grid search) z optymalizacją g-mean.
2. **Dodatkowe filtry** — np. *Black top-hat* (różnica między obrazem a jego zamknięciem morfologicznym) jako alternatywa lub uzupełnienie filtra Frangiego.
3. **Wieloskalowe podejście** — niezależna detekcja cienkich i grubych naczyń, potem łączenie.
4. **Klasyfikator ML** (cel na 4.0) — wyciąganie cech (wariancja, momenty, lokalne statystyki) z wycinków i trenowanie kNN / Random Forest / SVM.
5. **Sieć neuronowa** (cel na 5.0) — UNet uczony bezpośrednio na obrazach.

---

---

## 3b. Uczenie maszynowe — wymagania na 4.0

### 3b.1 Przygotowanie danych — wycinki i ekstrakcja cech

Po preprocessingu każdy obraz jest przetwarzany piksel po pikselu. Dla każdego piksela wyznaczamy wycinek **5×5 px** wyśrodkowany na tym pikselu (padding `reflect` przy krawędziach). Z wycinka obliczamy **13 cech**:

| # | Cecha | Opis |
|---|---|---|
| 1 | `mean` | Średnia intensywność pikseli wycinka |
| 2 | `std` | Odchylenie standardowe |
| 3 | `min` | Minimum |
| 4 | `max` | Maksimum |
| 5 | `range` | Rozstęp (max − min) |
| 6 | `mu20` | Moment centralny μ₂₀ (wariancja w kierunku x) |
| 7 | `mu02` | Moment centralny μ₀₂ (wariancja w kierunku y) |
| 8 | `mu11` | Moment centralny μ₁₁ (kowariancja x·y) |
| 9 | `mu30` | Moment centralny μ₃₀ (asymetria x) |
| 10 | `mu03` | Moment centralny μ₀₃ (asymetria y) |
| 11 | `H1` | Moment Hu H₁ = η₂₀ + η₀₂ (niezmiennik rotacji, ślad) |
| 12 | `H2` | Moment Hu H₂ = (η₂₀ − η₀₂)² + 4η₁₁² (anizotropia kształtu) |
| 13 | `vesselness` | Wartość filtra Frangiego w danym pikselu |

Etykietą każdej próbki jest wartość środkowego piksela maski eksperckiej (1 = naczynie, 0 = tło).

Momenty Hu obliczane są ze znormalizowanych momentów centralnych:
η_pq = μ_pq / (μ₀₀^(1+(p+q)/2))

Cecha `vesselness` pochodzi wprost z baseline'u — nie jest uczona, lecz podana klasyfikatorowi jako gotowa, silna cecha o kontekście przestrzennym.

### 3b.2 Wstępne przetwarzanie zbioru uczącego — undersampling

Naczynia stanowią ~7% pikseli FOV → zbiór surowy jest mocno niezrównoważony (~1:13). Zastosowano **ręczny undersampling**: dla każdego obrazu treningowego losujemy dokładnie **1 000 pikseli naczyniowych** i **1 000 pikseli tła** (z obszaru FOV). Dzięki temu zbór uczący jest idealnie zbalansowany (50/50), a klasyfikator nie jest obciążony w stronę klasy dominującej.

Łączny zbiór uczący: **20 obrazów × 2 000 próbek = 40 000 próbek**, 13 cech.

### 3b.3 Zastosowana metoda — Random Forest

Wybrany klasyfikator: **Random Forest** z biblioteki scikit-learn.

| Parametr | Wartość | Uzasadnienie |
|---|---|---|
| `n_estimators` | 100 | Kompromis dokładność / czas treningu |
| `max_depth` | 20 | Ogranicza przeuczenie; pełne drzewa dawały podobny wynik |
| `n_jobs` | -1 | Równoległy trening na wszystkich rdzeniach |
| `random_state` | 42 | Odtwarzalność |

**Dlaczego Random Forest?**
- Nie wymaga skalowania cech (cechy są różnych rzędów wielkości)
- Odporny na nadmiarowe cechy (np. momenty Hu mogą być słabo informatywne — RF je po prostu ignoruje)
- Szybki trening na 40 K próbkach
- Daje feature importances — można zweryfikować, które cechy są istotne

### 3b.4 Podział train/test (hold-out)

| Zbiór | Obrazy | Liczba obrazów |
|---|---|---|
| **Uczący** | Image_01L/R … Image_10L/R | 20 |
| **Testowy (hold-out)** | Image_11L/R … Image_14L/R | 8 |

Zbiór testowy jest identyczny z tym używanym w baseline, co umożliwia bezpośrednie porównanie metod.

### 3b.5 Wyniki wstępnej oceny — zbiór testowy hold-out

> **Uzupełnij po uruchomieniu notebooka (sekcja 11).**

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

### 3b.6 Porównanie: baseline vs Random Forest

| Metoda | Accuracy | Sensitivity | Specificity | G-mean | Bal. Acc. |
|---|---|---|---|---|---|
| Baseline (Frangi) | 0.9129 | 0.5760 | 0.9483 | 0.7381 | 0.7621 |
| Random Forest | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Delta (RF − Baseline) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 4. Wizualizacja wyników

Dla każdego analizowanego obrazu prezentujemy cztery widoki obok siebie:

1. **Oryginał** — kolorowa fotografia dna oka
2. **Ground truth** — maska ekspercka (czarno-biała, naczynia = białe)
3. **Predykcja** — wynik naszego algorytmu (czarno-biała maska)
4. **Nakładka** — fotografia z nałożoną kolorową analizą błędów:
   - 🟢 **zielony** = TP (true positive) — poprawnie wykryte naczynie
   - 🔴 **czerwony** = FP (false positive) — błędnie zaklasyfikowane jako naczynie (tło → naczynie)
   - 🔵 **niebieski** = FN (false negative) — przegapione naczynie (naczynie → tło)

```python
def make_overlay(rgb, pred, gt, fov_mask):
    overlay = rgb.copy()
    pred_m, gt_m = pred & fov_mask, gt & fov_mask
    overlay[pred_m & gt_m]   = [0, 255, 0]     # TP — zielony
    overlay[pred_m & ~gt_m]  = [255, 0, 0]     # FP — czerwony
    overlay[~pred_m & gt_m]  = [0, 0, 255]     # FN — niebieski
    return overlay
```

Wizualizacja pozwala szybko zlokalizować typy błędów: nadmiar czerwieni → algorytm "widzi naczynia tam, gdzie ich nie ma" (np. krawędzie tarczy nerwu wzrokowego, plamki); nadmiar niebieskiego → gubi cienkie naczynia.

W notebooku zaprezentowano wizualizacje dla 8 obrazów testowych (Image_11L do Image_14R) — zarówno dla metody baseline, jak i klasyfikatora Random Forest. Dla RF prezentujemy 5 kolumn: oryginał, ground truth, baseline, predykcja RF, nakładka RF.

---

## 5. Analiza wyników — metryki i interpretacja

### 5.1 Definicje miar

Dla problemu binarnego (naczynie = klasa pozytywna, tło = klasa negatywna) confusion matrix wygląda tak:

| | predykcja: naczynie | predykcja: tło |
|---|---|---|
| **ground truth: naczynie** | TP | FN |
| **ground truth: tło** | FP | TN |

Wyznaczane miary:

| Miara | Wzór | Interpretacja |
|---|---|---|
| Accuracy (trafność) | (TP+TN)/(TP+TN+FP+FN) | % poprawnie sklasyfikowanych pikseli |
| Sensitivity (czułość) | TP/(TP+FN) | jaki % naczyń algorytm wykrył |
| Specificity (swoistość) | TN/(TN+FP) | jaki % tła algorytm poprawnie odrzucił |
| G-mean | √(sens · spec) | średnia geometryczna — wrażliwa na niezrównoważenie klas |
| Balanced accuracy | (sens + spec) / 2 | średnia arytmetyczna — alternatywa dla g-mean |

**Dlaczego dwie miary "zrównoważone"?** Klasy są mocno niezrównoważone (~93% tło, ~7% naczynia). Naiwny klasyfikator "wszystko to tło" osiągnąłby accuracy = 93%, co wygląda świetnie, a jest bezużyteczne. Miary g-mean i balanced accuracy karzą takie zachowanie, bo czułość takiego klasyfikatora wynosi 0.

### 5.2 Wyniki baseline (Frangi) na zestawie testowym

| Obraz | Accuracy | Sensitivity | Specificity | G-mean | Bal. Acc. |
|---|---|---|---|---|---|
| Image_11L | 0.9070 | 0.6123 | 0.9320 | 0.7554 | 0.7721 |
| Image_11R | 0.9264 | 0.6935 | 0.9462 | 0.8100 | 0.8198 |
| Image_12L | 0.9097 | 0.5238 | 0.9567 | 0.7079 | 0.7402 |
| Image_12R | 0.9004 | 0.5255 | 0.9458 | 0.7050 | 0.7357 |
| Image_13L | 0.9079 | 0.5664 | 0.9419 | 0.7304 | 0.7542 |
| Image_13R | 0.9103 | 0.5434 | 0.9483 | 0.7178 | 0.7458 |
| Image_14L | 0.9298 | 0.6242 | 0.9641 | 0.7757 | 0.7941 |
| Image_14R | 0.9114 | 0.5187 | 0.9513 | 0.7025 | 0.7350 |
| **ŚREDNIA** | **0.9129** | **0.5760** | **0.9483** | **0.7381** | **0.7621** |

### 5.3 Wyniki Random Forest — hold-out

> **Uzupełnij po uruchomieniu notebooka.**

### 5.4 Interpretacja baseline

- **Accuracy ~91%** — wysoki wynik, ale pamiętajmy o niezrównoważeniu klas (naiwny model "wszystko = tło" miałby już ~93%).
- **Specificity ~95%** — algorytm bardzo dobrze odrzuca tło. Tylko ~5% pikseli tła generuje fałszywy alarm.
- **Sensitivity ~58%** — wykrywamy ok. 58% naczyń. Główne grube naczynia są łapane prawie w całości, natomiast cienkie naczynia obwodowe są często gubione (kontrast za mały, sygnał Frangiego nie przekracza progu).
- **G-mean ~0.74, Balanced acc ~0.76** — solidny wynik dla podejścia opartego wyłącznie na przetwarzaniu obrazu.

### 5.5 Typowe błędy algorytmu

**Fałszywe pozytywy (czerwony na nakładce):**
- Tarcza nerwu wzrokowego — krawędzie błędnie identyfikowane jako naczynie
- Krawędź FOV — cieniutka linia pozostała po erozji maski
- Refleksy świetlne, hiperpigmentacje

**Fałszywe negatywy (niebieski na nakładce):**
- Cienkie naczynia obwodowe — sygnał Frangiego zbyt słaby
- Naczynia w obszarach o niskim kontraście (np. plamka żółta)
- Rozwidlenia naczyń — geometria nie jest już idealnie tubularna

---

## 6. Podsumowanie

Zrealizowano wymagania na **3.0** (pipeline przetwarzania obrazu) oraz **4.0** (klasyfikator ML).

**Baseline (Frangi):** Na zestawie testowym (8 obrazów) uzyskano: accuracy = 0.91, sensitivity = 0.58, specificity = 0.95, g-mean = 0.74. Algorytm dobrze wykrywa grube naczynia, słabiej radzi sobie z cienkimi naczyniami obwodowymi i obszarami wokół tarczy nerwu wzrokowego.

**Random Forest:** Klasyfikator trenowany na 40 000 próbkach (20 obrazów, undersampling 1:1) z 13 cechami wyznaczonymi z wycinków 5×5 px. Wyniki liczbowe — po uzupełnieniu tabeli — pozwolą ocenić, o ile klasyfikator ML poprawia (lub nie) wyniki baseline, szczególnie w zakresie czułości na cienkie naczynia.

---

## Literatura

- A. F. Frangi et al., *Multiscale vessel enhancement filtering*, MICCAI 1998.
- Baza CHASE_DB1: https://blogs.kingston.ac.uk/retinal/chasedb1/
- Dokumentacja scikit-image: https://scikit-image.org/
- Dokumentacja scikit-learn: https://scikit-learn.org/
- P. Liskowski, K. Krawiec, *Segmenting Retinal Blood Vessels With Deep Neural Networks*, IEEE TMI 2016.
