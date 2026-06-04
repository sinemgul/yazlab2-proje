# Proje raporu şablonu (Word/PDF’e aktarın)

Bu dosya **hazır rapor değildir**; tam deney sonrası doldurmanız için iskelet.
Sayıları `artifacts/results/experiment_report.md` dosyasından alın.

---

## 1. Kapak

- Ders: Yazılım Laboratuvarı II  
- Proje başlığı: From Black-Box to Explainability (veya PDF başlığı)  
- Ekip: Sinem Gül, Elif Aysan  
- Tarih  

## 2. Özet (Abstract)

3–5 cümle: ne yaptınız, hangi veri setleri, otomata vs DL karşılaştırması,
en önemli bulgu (ör. hangi senaryoda hangi model önde).

## 3. Giriş

- Zaman serisi anomali tespiti problemi  
- Black-box vs açıklanabilir model motivasyonu  
- BATADAL ve SKAB kısa tanımı  
- Projenin amacı ve kapsamı  

## 4. İlgili çalışmalar

- SAX / PAA (Lin vd.)  
- Olasılıksal otomata / sembolik yaklaşım  
- LSTM, GRU, 1D-CNN ile zaman serisi sınıflandırma  
- (2–3 kaynak, IEEE/dergi formatında)  

## 5. Yöntem

### 5.1 Veri ve ön işleme

- BATADAL: %60 / %20 / %20 zaman sıralı bölme  
- SKAB: `source_file` GroupKFold  
- Eksik veri, normalizasyon, PCA (yalnızca train’de fit)  

### 5.2 Olasılıksal otomata

- PAA → SAX → sliding window  
- Geçiş olasılıkları, Laplace smoothing  
- Unseen: train SAX sözlüğü + Levenshtein (test verisi değiştirilmez)  
- Açıklanabilirlik alanları (state, pattern, path probability, …)  

### 5.3 Derin öğrenme

- LSTM, GRU, 1D-CNN; sekans uzunluğu, epoch, early stopping  

### 5.4 Deney tasarımı

- 5 seed, 3 senaryo (original, gürültü, unseen)  
- Otomata parametre taraması (window, alphabet)  
- Metrikler: F1, Precision, Recall; ROC/PR; Detection Rate, Mapping Accuracy  
- İstatistik: Wilcoxon, McNemar  
- Cross-dataset: BATADAL ↔ SKAB  

## 6. Deneysel sonuçlar

Aşağıdaki tabloları `experiment_report.md`’den kopyalayın ve **yorumlayın**
(sadece tablo yapıştırmak yetmez).

| Kaynak | İçerik |
|--------|--------|
| Tablo 1 | Model × veri seti F1 ± std |
| Tablo 2 | Gürültü etkisi, Detection Rate, Mapping Accuracy |
| Tablo 3 | Cross-dataset matrisi |
| Tablo 4 | Window / alphabet duyarlılığı |
| Tablo 5 | Eğitim / çıkarım süreleri |

Örnek yorum soruları:

- Gürültü hangi modeli daha çok düşürüyor?  
- Unseen’de otomata Levenshtein ne kadar işe yarıyor?  
- DL otomata’ya göre F1’de anlamlı fark var mı? (Wilcoxon p-değeri)  
- Cross-dataset sonuçları genellenebilirlik hakkında ne söylüyor?  

### Şekiller (en az 3 önerilir)

1. Örnek confusion matrix (BATADAL veya SKAB)  
2. ROC veya PR eğrisi  
3. Otomata transition heatmap veya state diagram  
4. (İsteğe bağlı) sensitivity ısı haritası  

Dosyalar: `artifacts/figures/*.png`

## 7. Tartışma

- Açıklanabilirlik avantajı / maliyeti  
- Otomata parametreleri (state sayısı, geçiş yoğunluğu)  
- Sınırlamalar (veri boyutu, CPU süresi, SKAB fold sayısı)  

## 8. Sonuç

- Ana bulgular (madde madde)  
- Gelecek iş (GPU, daha fazla model, gerçek zamanlı SCADA)  

## 9. Kaynakça

APA veya IEEE; en az 5 kaynak.

## Ek A — Örnek açıklanabilirlik çıktısı

`artifacts/explanations/*.jsonl` dosyasından 1–2 adım örneği (JSON veya tablo).
