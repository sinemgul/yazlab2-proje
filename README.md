# Yazlab2 — Olasılıksal Otomata vs Derin Öğrenme

**Yazılım Laboratuvarı II** ders projesi. BATADAL ve SKAB zaman serilerinde
black-box modeller (LSTM, GRU, 1D-CNN) ile açıklanabilir **olasılıksal otomata**
yaklaşımını aynı pipeline üzerinde karşılaştırıyoruz.

| | |
|---|---|
| **Ekip** | Sinem Gül, Elif Aysan |
| **Repo** | https://github.com/sinemgul/yazlab2-proje |

## Proje ne yapıyor?

Veriyi okuyup ön işliyoruz, train/val/test ayırıyoruz, sonra hem derin öğrenme
hem SAX + otomata ile sınıflandırma yapıyoruz. Metrikler, grafikler ve otomata
için adım adım açıklamalar `artifacts/` klasörüne yazılıyor. Parametrelerin
çoğu `src/config.py` dosyasında; PDF’teki deney tasarımına göre ayarladık.

## Kurulum

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`torch` kurulu değilse DL kısmı atlanır, otomata tarafı yine çalışır.

## Veri yolları

Kendi makinenize göre `src/config.py` içindeki yolları düzenleyin. Bizim
örnek:

- BATADAL: `Downloads\BATADAL_dataset04.csv`
- SKAB: `Downloads\archive (3)\SKAB`

## Çalıştırma

Tam deney (uzun sürebilir, saatler alabilir):

```powershell
python -m src.main
```

Hızlı deneme / figür üretimi:

```powershell
python scripts/smoke_pipeline.py
```

Grafiklere bakmak için tarayıcıda açın:
`artifacts/smoke/figures/gallery.html` (smoke sonrası oluşur).

## Çıktılar (`artifacts/`)

- `results/` — csv, jsonl, `experiment_report.md` (tablolar)
- `figures/` — confusion matrix, ROC/PR, heatmap, state diagram
- `explanations/` — otomata adımları (json/jsonl)
- `logs/` — koşum logu ve config snapshot

## Testler

```powershell
pytest
```

## Kod yapısı (özet)

```
src/
  config.py          parametreler
  data/              BATADAL, SKAB, ön işleme, split
  automata/          SAX, otomata, açıklanabilirlik
  models/            LSTM, GRU, 1D-CNN
  evaluation/        metrikler, grafikler
  experiments/       senaryolar (original / gürültü / unseen)
  pipeline/          uçtan uca koşum
tests/               birim testler
```

## Deney notları (PDF özeti)

- 5 farklı seed, 3 senaryo (original, Gaussian noise, unseen)
- DL: LSTM, GRU, 1D-CNN — otomata için window/alphabet sweep
- BATADAL: zaman sıralı %60 / %20 / %20
- SKAB: `source_file` ile GroupKFold (sızıntı olmasın diye scaler/SAX sadece train’de fit)
- **Unseen:** test verisini bozmuyoruz; eğitimde görülmeyen SAX pattern’ları
  sözlükte yoksa unseen sayılıp Levenshtein ile eşleniyor

Detaylı tablo ve istatistik testleri koşum bitince
`artifacts/results/experiment_report.md` dosyasında üretiliyor.

## Lisans

Eğitim amaçlı ders çalışması; veri setleri kendi lisanslarına tabidir.
