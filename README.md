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

pytest -q

```



`torch` kurulu değilse DL kısmı atlanır, otomata tarafı yine çalışır.



## Veri yolları



Kendi makinenize göre `src/config.py` içindeki yolları düzenleyin. Bizim

örnek:



- BATADAL: `Downloads\BATADAL_dataset04.csv`

- SKAB: `Downloads\archive (3)\SKAB`



## Çalıştırma



**Tam deney** (PDF matrisi; saatler sürebilir):



```powershell

python -m src.main

```



Sonuçlar:



| Dosya / klasör | Açıklama |

|----------------|----------|

| `artifacts/results/experiment_report.md` | Tablo 1–5 + istatistikler |

| `artifacts/figures/*.png` | Confusion, ROC/PR, heatmap, sensitivity |

| `artifacts/figures/gallery.html` | Tüm figürler (otomatik üretilir) |

| `artifacts/dashboard.html` | Tablolar + örnek figürler + galeri linki (tek sayfa) |

| `artifacts/explanations/*.jsonl` | Otomata adım açıklamaları |



**Hızlı smoke test** (~3 dk):



```powershell

python scripts/smoke_pipeline.py

```



Smoke çıktıları `artifacts/smoke/` altında; galeri:

`artifacts/smoke/figures/gallery.html`



Galeriyi elle yenilemek için:



```powershell

python scripts/build_gallery.py

python scripts/build_dashboard.py

```



## Teslim dokümanları



| Dosya | Ne için |

|-------|---------|

| [docs/TESLIM_CHECKLIST.md](docs/TESLIM_CHECKLIST.md) | Teslim öncesi kontrol listesi |

| [docs/RAPOR_SABLON.md](docs/RAPOR_SABLON.md) | Word/PDF rapor iskeleti (siz doldurursunuz) |

| [docs/SUNUM_TASLAGI.md](docs/SUNUM_TASLAGI.md) | Sunum slayt başlıkları |



`artifacts/` git’e girmez (`.gitignore`); rapor figürlerini Word/PDF’e kopyalayın.



## Kod yapısı (özet)



```

src/

  config.py          parametreler

  data/              BATADAL, SKAB, ön işleme, split

  automata/          SAX, otomata, açıklanabilirlik

  models/            LSTM, GRU, 1D-CNN

  evaluation/        metrikler, grafikler, galeri

  experiments/       senaryolar, rapor üretici

  pipeline/          uçtan uca koşum

tests/               birim testler

docs/                teslim şablonları

scripts/             smoke, build_gallery

```



## Deney notları (PDF özeti)



- 5 seed, 3 senaryo (original, Gaussian noise, unseen)

- DL: LSTM, GRU, 1D-CNN — otomata window/alphabet sweep

- BATADAL: %60 / %20 / %20 zaman sıralı

- SKAB: GroupKFold; scaler/SAX yalnızca train’de fit

- Unseen: test verisi değişmez; train SAX sözlüğü dışı pattern’lar Levenshtein ile eşlenir



## Lisans



Eğitim amaçlı ders çalışması; veri setleri kendi lisanslarına tabidir.

