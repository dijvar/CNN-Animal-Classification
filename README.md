# CNN Görüntü Sınıflandırma Projesi

Bu proje, CNN kullanarak görüntü sınıflandırma yapan ve farklı ışık koşullarında modelin performansını test eden bir uygulamadır. Aygaz Görüntü İşleme Bootcamp projesi kapsamında geliştirilmiştir.

***NOT***: Projeye ait notebook Colab üzerinde çalıştırılmıştır. Notebook Kaggle'a yüklenmiştir ancak çalıştırılmamıştır. 

## Links

- [Kaggle](https://www.kaggle.com/code/orhandjvarekinci/cnn-animal-classification-grid-search)
- [WandB](https://wandb.ai/orhandijvar/bootcamp-cnn-animal-classification?nw=nwuserorhandijvar)
- [Google Colab](https://colab.research.google.com/drive/1D2SisvJgJ8DjERSBMBCEanhvwlsmmGjr?usp=sharing)
- [Cursor](https://www.cursor.com/)

## Veri Seti ve Özellikleri

Proje, 10 farklı hayvan sınıfını (collie, dolphin, elephant, fox, moose, rabbit, sheep, squirrel, giant panda, polar bear) sınıflandırmak için tasarlanmıştır. Her sınıftan 650 görüntü kullanılarak dengeli bir veri seti oluşturulmuştur.

### Kullanılan Temel Kütüphaneler ve Araçlar

- **TensorFlow/Keras**: CNN modelinin oluşturulması ve eğitimi
- **OpenCV**: Görüntü işleme ve manipülasyon
- **NumPy**: Matris işlemleri ve veri manipülasyonu
- **Scikit-learn**: Veri seti bölme ve model değerlendirme
- **Wandb**: Deney takibi ve hiperparametre optimizasyonu
- **[Google Colab](https://colab.research.google.com/drive/1D2SisvJgJ8DjERSBMBCEanhvwlsmmGjr?usp=sharing)**: GPU destekli, bulut tabanlı Jupyter Notebook ortamı
- [Cursor](https://www.cursor.com/): Yapay Zeka ile kodlama için IDE

### Teknik Detaylar

- **Veri Ön İşleme**:

  - Görüntü boyutu normalizasyonu (224x224 ve 128x128)
  - Piksel değerlerinin [0,1] aralığına normalizasyonu
  - Veri artırma teknikleri (rotasyon, zoom, parlaklık ayarı)
- **Model Mimarisi**:

  - Çoklu konvolüsyon ve havuzlama katmanları
  - Batch normalizasyon
  - Dropout ile regularizasyon
  - Farklı aktivasyon fonksiyonları (ReLU, PReLU, ELU)
- **Hiperparametre Optimizasyonu**:

  - Grid Search kullanılarak en iyi parametre kombinasyonunun bulunması
  - Optimize edilen parametreler:
    - Giriş boyutu
    - Aktivasyon fonksiyonu
    - Öğrenme oranı
    - Veri artırma stratejileri

## Özellikler

- Veri seti hazırlama ve bölme (Train, Validation, Test)
- Görüntü manipülasyonu ve renk sabitliği işlemleri
- Grid Search ile model parametrelerinin optimizasyonu
- Wandb entegrasyonu ile deney takibi
- Modüler ve nesne yönelimli kod yapısı

### Grid Search Tercihi

Bu projede hiperparametre optimizasyonu için Grid Search tercih edilmiştir. Bayesian Optimization, Random Search gibi daha gelişmiş yöntemler mevcut olsa da, bootcamp kapsamında öğrencilerin hiperparametrelerin model performansına etkisini daha iyi gözlemleyebilmeleri için Grid Search kullanılmıştır. Bu sayede her parametre kombinasyonunun sonuçları sistematik olarak incelenebilmektedir.

#### Grid Search Parametreleri

Aşağıdaki parametreler için Grid Search uygulanmıştır:

1. **Giriş Boyutları**:

   - 224x224 piksel
   - 128x128 piksel
2. **Aktivasyon Fonksiyonları**:

   - ReLU (Rectified Linear Unit)
   - PReLU (Parametric ReLU)
   - ELU (Exponential Linear Unit)
3. **Öğrenme Oranları**:

   - 0.01
   - 0.001
   - 0.0001
4. **Veri Artırma Stratejileri**:

   - Aktif ve Pasif durumlar test edilmiştir
   - Aktif durumda kullanılan parametreler:
     - Rotasyon: ±15 derece
     - Genişlik/Yükseklik Kaydırma: ±%12
     - Yatay Çevirme: Aktif
     - Zoom: ±%10
     - Parlaklık Aralığı: [0.9, 1.1]
     - Kesme: 10 derece
     - Kanal Kaydırma: ±0.1

Toplam Deney Sayısı: 2 (giriş boyutu) × 3 (aktivasyon) × 3 (öğrenme oranı) × 2 (veri artırma) = 36 farklı kombinasyon

**Sabit Tutulan Parametreler:**

- Batch Size: 64
- Epoch Sayısı: 100
- Optimizer: Adam
- Kayıp Fonksiyonu: Sparse Categorical Crossentropy
- Dropout Oranı: 0.2
- Weight Decay: 0.0001

## Dosya Açıklamaları

- `src/model.py`: CNN modelinin tanımlandığı ve Grid Search implementasyonunun yapıldığı modül
- `src/dataset.py`: Veri seti işleme, bölme ve artırma işlemlerinin yapıldığı modül
- `src/utils.py`: Renk manipülasyonu ve renk sabitliği algoritmalarının implementasyonları
- `src/train.py`: Model eğitimi ve değerlendirme işlemlerinin yapıldığı ana modül
- `src/config.py`: Konfigürasyon sınıfları ve YAML dosya yönetimi
- `config/config.yaml`: Proje parametrelerinin tanımlandığı konfigürasyon dosyası

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

```
tensorflow==2.15.0
numpy>=1.19.2
opencv-python>=4.5.1
PyYAML>=5.4.1
scikit-learn>=0.24.2
wandb>=0.12.0
matplotlib>=3.3.4
tqdm>=4.61.0 
```

```bash
pip install -r requirements.txt
```

## Proje Yapısı

```
.
├── config/
│   └── config.yaml         # Proje konfigürasyonu
├── dataset/               # Ham veri seti
├── processed_dataset/     # İşlenmiş veri seti
├── src/
│   ├── __init__.py
│   ├── config.py         # Konfigürasyon sınıfı
│   ├── dataset.py        # Veri seti işleme
│   ├── model.py          # CNN model tanımı
│   ├── train.py          # Eğitim kodu
│   └── utils.py          # Yardımcı fonksiyonlar
├── checkpoints/          # Model ağırlıkları
├── logs/                 # Eğitim logları
├── results/             # Test sonuçları
├── requirements.txt     # Bağımlılıklar
└── README.md
```

## Kullanım

1. Veri setini `dataset/` klasörüne yerleştirin
2. `config/config.yaml` dosyasını düzenleyin
3. Wandb hesap bilgilerinizi güncelleyin
4. Eğitimi başlatın:

```bash
python -m src.train
```

## Konfigürasyon

`config.yaml` dosyasında aşağıdaki parametreleri düzenleyebilirsiniz:

- Veri seti bölme oranları
- Model parametreleri (input size, aktivasyon fonksiyonu, learning rate)
- Batch size ve epoch sayısı
- Dropout oranı
- Wandb proje ayarları

## Sonuçlar

Eğitim sonuçları ve model performansı [Wandb](https://wandb.ai/orhandijvar/bootcamp-cnn-animal-classification?nw=nwuserorhandijvar) üzerinden takip edilebilir. Her deney için aşağıdaki metrikler kaydedilir:

- Eğitim ve validasyon kayıpları/doğrulukları
- Test sonuçları (orijinal, manipüle edilmiş ve renk sabitliği uygulanmış görüntüler için)
- Model parametreleri ve hiperparametreler
