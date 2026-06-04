# Sunum taslağı (10–15 dk)

Her slayt için notları kendi deney sonuçlarınıza göre doldurun.

---

### Slayt 1 — Başlık

- Proje adı, ekip, ders  
- Repo linki: https://github.com/sinemgul/yazlab2-proje  

### Slayt 2 — Problem

- SCADA / su şebekesi zaman serilerinde anomali  
- Black-box modellerin yorumlanamaması  

### Slayt 3 — Amaç

- DL (LSTM, GRU, 1D-CNN) vs olasılıksal otomata  
- Aynı pipeline, adil karşılaştırma  

### Slayt 4 — Veri setleri

- BATADAL Training Dataset 2  
- SKAB valve1 + valve2  
- (1 küçük zaman serisi grafiği varsa iyi olur)  

### Slayt 5 — Pipeline şeması

```
Veri → ön işleme → split → [DL | SAX+Otomata] → metrikler → rapor
```

### Slayt 6 — Otomata (kısa)

- PAA / SAX / sliding window  
- Unseen: sözlük + Levenshtein  
- **Canlı veya ekran görüntüsü:** bir `explanations/*.jsonl` satırı  

### Slayt 7 — Derin öğrenme

- Üç model, sekans penceresi, early stopping  

### Slayt 8 — Deney tasarımı

- 5 seed, 3 senaryo, parametre sweep  
- Cross-dataset  

### Slayt 9 — Tablo 1 özeti

- En iyi F1 hangi model / veri setinde? (1 tablo, büyük font)  

### Slayt 10 — Gürültü ve unseen

- Tablo 2’den 1–2 satır  
- Detection Rate / Mapping Accuracy ne anlama geliyor?  

### Slayt 11 — Otomata vs DL

- Wilcoxon / McNemar sonucu (p-değeri)  
- 1 ROC veya confusion matrix figürü  

### Slayt 12 — Cross-dataset

- Tablo 3 — transfer zor mu?  

### Slayt 13 — Süre / ölçeklenebilirlik

- Tablo 5 — otomata hızlı mı, DL yavaş mı?  

### Slayt 14 — Sonuç

- 3 madde bulgu  
- Sınırlama + gelecek iş  

### Slayt 15 — Sorular

- Teşekkür  

---

## Demo önerisi (1–2 dk)

1. `artifacts/figures/gallery.html` veya tek bir heatmap PNG  
2. `explanations` JSONL’den bir adım: state, pattern, decision  

## Hazırlık

- Tam deney bitmeden sayıları slayta yazmayın; son gün güncelleyin.  
- Yedek: smoke galerisi offline açılabilir (internet gerekmez).
