# Hareket Algılama ve Öğrenme Sistemi

Bu proje, video görüntülerinde hareket algılama yapan ve makine öğrenmesi ile bu hareketleri sınıflandırabilen bir sistem içerir.

## Gereksinimler

- Python 3.10.9
- OpenCV
- NumPy
- scikit-learn
- imutils

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Video dosyanızı projenin ana dizinine kopyalayın.

## Kullanım

Programı çalıştırmak için:

```bash
python motion_detector.py
```

### Kontroller

- 'q' tuşu: Programdan çıkış
- 't' tuşu: Toplanan verileri kullanarak modeli eğit

### Sistem Özellikleri

1. Gerçek zamanlı hareket algılama
2. Otomatik eğitim veri seti oluşturma
3. Makine öğrenmesi ile hareket sınıflandırma
4. Model kaydetme ve yükleme özelliği

### Çalışma Mantığı

1. Program başlatıldığında, video kaynağından görüntüler alınır
2. Her frame'de hareket algılama yapılır
3. Algılanan hareketler yeşil dikdörtgenler ile işaretlenir
4. Her 30 frame'de bir otomatik olarak eğitim verisi toplanır
5. 't' tuşuna basıldığında toplanan veriler ile model eğitilir

## Özelleştirme

`MotionDetector` sınıfının parametrelerini değiştirerek sistem hassasiyetini ayarlayabilirsiniz:

- `min_area`: Algılanacak minimum hareket alanı
- `history`: Arka plan çıkarıcının kullanacağı geçmiş frame sayısı
- `varThreshold`: Hareket algılama hassasiyeti 