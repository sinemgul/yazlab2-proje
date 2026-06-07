# Yazlab2 — Olasılıksal Otomata vs Derin Öğrenme

**Yazılım Laboratuvarı II** ders projesi. BATADAL ve SKAB zaman serilerinde black-box modeller (LSTM, GRU, 1D-CNN) ile açıklanabilir **olasılıksal otomata** yaklaşımını aynı pipeline üzerinde karşılaştırıyoruz.

| | |
|---|---|
| **Ekip** | Sinem Gül, Elif Aysan |
| **Repo** | https://github.com/sinemgul/yazlab2-proje |

## Ödev ne?

Endüstriyel zaman serilerinde **anomali tespiti** yapıyoruz. İki yaklaşımı aynı veri ve aynı metriklerle karşılaştırmamız istendi:

- **Derin öğrenme** (LSTM, GRU, 1D-CNN) — yüksek doğruluk, ama karar süreci black-box
- **Olasılıksal otomata** (PAA + SAX + geçiş olasılıkları) — daha düşük doğruluk beklentisi, ama adım adım açıklanabilir

İki veri seti: **BATADAL** (su dağıtım SCADA) ve **SKAB** (endüstriyel vana sensörleri). Deneyler 5 seed, 3 senaryo (original, gürültü, unseen), parametre taraması, cross-dataset değerlendirme ve istatistiksel testlerle yürütüldü.

## Ne yaptık?

Uçtan uca bir Python pipeline kurduk:

1. Veri yükleme ve ön işleme (eksik doldurma, standardizasyon, otomata için PCA)
2. Train / val / test bölme (BATADAL zaman sıralı; SKAB GroupKFold)
3. Dört model eğitimi ve değerlendirme (automata, LSTM, GRU, 1D-CNN)
4. Metrikler: F1, Precision, Recall, ROC/PR, Detection Rate, Mapping Accuracy
5. Otomata için jsonl açıklama çıktıları, heatmap, state diagram
6. Wilcoxon / McNemar istatistiksel testleri
7. Otomatik rapor, galeri ve dashboard üretimi

Tam deney matrisi **1716 koşu** ile tamamlandı. Sonuçlar `artifacts/` altında.

### Öne çıkan bulgular

| | BATADAL F1 | SKAB F1 |
|---|------------|---------|
| automata | 0,10 | 0,03 |
| GRU | 0,18 | **0,87** |
| LSTM | 0,12 | **0,87** |
| 1D-CNN | 0,00 | 0,86 |

SKAB'ta derin modeller belirgin üstün; otomata düşük F1 ama her adımı açıklayabiliyor. Cross-dataset genelleme zayıf (F1 ≈ 0,04–0,09).

## Kurulum

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

`torch` kurulu değilse DL kısmı atlanır, otomata tarafı yine çalışır.

Veri yollarını `src/config.py` içinde kendi makinenize göre ayarlayın:

- BATADAL: `Downloads\BATADAL_dataset04.csv`
- SKAB: `Downloads\archive (3)\SKAB`

## Çalıştırma

**Tam deney** (saatler sürebilir):

```powershell
python -m src.main
```

**Hızlı smoke test** (~3 dk):

```powershell
python scripts/smoke_pipeline.py
```

Dashboard ve galeriyi yenilemek için:

```powershell
python scripts/build_gallery.py
python scripts/build_dashboard.py
```

### Çıktılar

| Dosya / klasör | Açıklama |
|----------------|----------|
| `artifacts/results/experiment_report.md` | Tablo 1–5 + istatistikler |
| `artifacts/dashboard.html` | Tüm tablolar, 548 figür, dosya linkleri |
| `artifacts/figures/gallery.html` | Figür galerisi |
| `artifacts/explanations/*.jsonl` | Otomata adım açıklamaları |

## Kod yapısı

```
src/
  config.py          parametreler
  data/              BATADAL, SKAB, ön işleme, split
  automata/          SAX, otomata, açıklanabilirlik
  models/            LSTM, GRU, 1D-CNN
  evaluation/        metrikler, grafikler, dashboard
  experiments/       senaryolar, rapor üretici
  pipeline/          uçtan uca koşum
tests/               birim testler
scripts/             smoke, galeri, dashboard
```

## Deney tasarımı (kısa)

- 5 seed: 42, 7, 123, 2026, 999
- 3 senaryo: original, Gaussian noise (std=0,1), unseen (Levenshtein eşleme)
- Otomata parametre taraması: pencere {3,4,5,6}, alfabe {3,4,5,6}
- BATADAL: %60 / %20 / %20 zaman sıralı bölme
- SKAB: 5-fold GroupKFold (`source_file`); scaler ve SAX yalnızca train'de fit
