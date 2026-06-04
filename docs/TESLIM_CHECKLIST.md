# Teslim öncesi kontrol listesi

Tam deney ve yazılı raporu siz koşturacaksınız; aşağıdaki maddelerin geri kalanı
repo tarafında hazır.

## Kod ve repo

- [ ] `pytest` hatasız (`pytest -q`)
- [ ] `src/config.py` veri yolları bu bilgisayarda doğru
- [ ] GitHub güncel: https://github.com/sinemgul/yazlab2-proje

## Tam deney (sizin yapacağınız)

```powershell
python -m src.main
```

Bittiğinde kontrol edin:

- [ ] `artifacts/results/experiment_report.md` oluştu
- [ ] `artifacts/results/statistical_tests.csv` dolu
- [ ] `artifacts/figures/` altında PNG’ler var
- [ ] `artifacts/figures/gallery.html` tarayıcıda açılıyor
- [ ] `artifacts/logs/config_snapshot.json` kayıtlı

Sorun olursa log: `artifacts/logs/yazlab2.experiments.log`

## Yazılı rapor (sizin yapacağınız)

Şablon: `docs/RAPOR_SABLON.md`

- [ ] Tablo 1–5 ve istatistikleri `experiment_report.md`’den aktardınız
- [ ] Seçtiğiniz 3–5 figürü rapora eklediniz (`artifacts/figures/`)
- [ ] Giriş, yöntem, bulgular, sonuç, kaynakça yazıldı
- [ ] PDF/Word teslim formatına uygun

## Sunum (8–12 Haziran)

Taslak slayt başlıkları: `docs/SUNUM_TASLAGI.md`

- [ ] 10–15 dakikalık slayt
- [ ] Demo: repo + bir confusion matrix / otomata açıklama örneği
- [ ] Ekip payı net (kim ne yaptı)

## Not

`artifacts/` klasörü `.gitignore` içinde; büyük çıktılar repoya gitmez.
Rapor için figürleri Word/PDF’e kopyalayın veya sunumda yerel klasörden açın.
