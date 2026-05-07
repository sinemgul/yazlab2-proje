# Yazlab2 — From Black-Box to Explainability

Zaman serisi verileri üzerinde **derin öğrenme tabanlı (black-box)** ve **olasılıksal otomata tabanlı (interpretable)** modelleri sistematik biçimde karşılaştıran proje. PDF'te tanımlanan tüm mimari, deney ve raporlama gereksinimleri tek bir merkezi konfigürasyon ve modüler pipeline üzerinden yönetilir.

## Proje Yapısı

```
src/
  config.py                  Tüm parametrelerin merkezi tanımı (PDF Bölüm VIII)
  data/
    batadal.py               BATADAL Training Dataset 2 yükleyicisi
    skab.py                  SKAB valve1 + valve2 birleştirici (source_group/source_file)
    preprocessing.py         Eksik veri, normalizasyon, PCA, gürültü
    splits.py                Time-ordered + GroupKFold/StratifiedGroupKFold
  automata/
    sax.py                   PAA + SAX + sliding window
    levenshtein.py           Edit distance + en yakın pattern eşleme
    automaton.py             Olasılıksal otomata + path probability + güven skoru
    explainability.py        JSON/JSONL açıklanabilirlik çıktıları + transition matrix
    counterfactual.py        Counterfactual analiz (PDF X.D opsiyonel ek puan)
  models/
    deep_learning.py         LSTM / GRU / 1D-CNN sınıflandırıcılar (PyTorch) + süre ölçümü
    sequence_dataset.py      Sliding-window sekans üretimi
  evaluation/
    metrics.py               Accuracy/Precision/Recall/F1 + ROC/PR + Wilcoxon + McNemar
    visualization.py         Confusion matrix, transition heatmap, state diagram, sensitivity
  experiments/
    scenarios.py             Original / Gaussian noise / Unseen senaryoları
    automata_runner.py       Otomata değerlendirme + Detection Rate + Mapping Accuracy
    report.py                Markdown rapor üretici (Tablo 1-5)
  pipeline/
    runner.py                Uçtan uca BATADAL + SKAB + cross-dataset + istatistik testleri
  utils/
    logging.py               JSONL loglama + JSON snapshot
    seeding.py               Tek noktadan seed kontrolü
  main.py                    `python -m src.main` giriş noktası
tests/                       PyTest birim testleri (Levenshtein, SAX, otomata, splits, metrics)
```

## Kurulum

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Not: `torch` kurulumu işletim sistemi/GPU sürümüne göre değişebilir. PyTorch
> mevcut değilse pipeline derin öğrenme adımlarını otomatik atlar; otomata
> tarafı bağımsız çalışmaya devam eder.

## Veri Yolları

`src/config.py` içindeki `BatadalConfig.csv_path` ve `SkabConfig.root_dir`
değerlerini sisteminizdeki konumlara göre güncelleyin. Varsayılan yollar
şöyledir:

- BATADAL: `c:\Users\sinem\Downloads\BATADAL_dataset04.csv`
- SKAB: `c:\Users\sinem\Downloads\archive (3)\SKAB`

## Çalıştırma

```powershell
python -m src.main
```

Çıktılar `artifacts/` altında oluşturulur:

- `artifacts/results/<dataset>_<model>_runs.jsonl` — her deneyin satır bazlı kaydı (metrikler, süreler, ROC/PR, Detection Rate, Mapping Accuracy)
- `artifacts/results/<dataset>_<model>_summary.csv` — senaryo/parametre bazlı ortalama ± std (süreler ve skorlar dahil)
- `artifacts/results/cross_dataset_runs.jsonl` — BATADAL ↔ SKAB transfer sonuçları
- `artifacts/results/statistical_tests.csv` — Wilcoxon + McNemar sonuçları (otomata/DL çiftleri)
- `artifacts/results/experiment_report.md` — Rapor için 5 tablo + istatistik testleri (Markdown)
- `artifacts/results/*_transitions.csv` — geçiş olasılıkları matrisleri
- `artifacts/results/predictions/*.npz` — istatistik testleri için cache'lenmiş tahminler
- `artifacts/explanations/*.jsonl` — olasılıksal açıklanabilirlik kayıtları
- `artifacts/explanations/counterfactual_*.json` — counterfactual analiz örneklemleri
- `artifacts/figures/cm_*.png` — confusion matrix figürleri
- `artifacts/figures/roc_*.png`, `pr_*.png` — ROC ve Precision-Recall eğrileri
- `artifacts/figures/heatmap_*.png` — geçiş olasılığı heatmap'ları
- `artifacts/figures/diagram_*.png` — automata state diagramları
- `artifacts/figures/sensitivity_*.png` — window/alphabet duyarlılık ısı haritası
- `artifacts/logs/yazlab2.experiments.log` — koşum logu
- `artifacts/logs/config_snapshot.json` — kullanılan tam konfigürasyon

## Testler

```powershell
pytest
```

Testler aşağıdaki davranışları doğrular:

- `tests/test_levenshtein.py` — Levenshtein doğruluğu, en yakın pattern eşleme,
  boş aday listesi davranışı, deterministik tie-break (PDF Bölüm VI gereksinimi).
- `tests/test_sax.py` — PAA/SAX doğruluğu, sliding window sayısı, alfabe haritası.
- `tests/test_automaton.py` — Geçiş olasılıkları, unseen pattern yönetimi, path
  probability ve karar eşiği.
- `tests/test_splits.py` — Time-ordered bölme, GroupKFold sızıntı kontrolü.
- `tests/test_metrics.py` — Accuracy/F1, Wilcoxon, McNemar yardımcıları.

## Deney Tasarımı (PDF Bölüm VII ile birebir)

| Bileşen | Değer |
|---|---|
| Random seedler | 42, 123, 2026, 7, 999 |
| Senaryolar | original, Gaussian noise, unseen |
| Sabit otomata parametreleri | window=4, alphabet=3 |
| Sweep | window ∈ {3,4,5,6}, alphabet ∈ {3,4,5,6} |
| BATADAL split | %60 / %20 / %20, zaman sıralı |
| SKAB split | `source_file` üzerinde GroupKFold / StratifiedGroupKFold |
| Epoch üst sınırı | 50 |
| Batch size | 32 |
| Early stopping | val_loss, patience=5 |

Veri sızıntısı önleme: scaler/PCA/SAX sözlüğü yalnızca eğitim kümesi
üzerinde fit edilir (`src/data/preprocessing.py`,
`src/automata/automaton.py`).

## Açıklanabilirlik (PDF Bölüm X)

Her test adımı için `state`, `pattern`, `status`, `mapped_to`, `transitions`,
`path probability`, `decision` ve `confidence` alanlarını içeren JSON kayıt
üretilir (`src/automata/explainability.py`). PDF örnek formatı korunmuştur.

## Lisans

Eğitim amaçlı, ders projesi olarak hazırlanmıştır.
