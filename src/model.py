from typing import Tuple, Dict, Any
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from pathlib import Path
import os
from config import Config


class CNNModel:
    # Sınıf değişkeni olarak deney sayacını tanımla
    _current_experiment_num = 0
    _best_experiment = {
        'name': None,
        'accuracy': 0.0,
        'config': None,
        'results': None
    }
    
    def __init__(self, config: Config, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.current_experiment: Dict[str, Any] = {}
        
        # Toplam deney sayısını hesapla
        self.total_experiments = (
            len(self.config.model.input_sizes) *
            len(self.config.model.activations) *
            len(self.config.model.learning_rates) *
            len(self.config.model.data_augmentation.enabled)  # Data augmentation için
        )
        
        print(f"\nToplam deney sayısı: {self.total_experiments}")
    
    @classmethod
    def get_next_experiment_num(cls) -> int:
        """Bir sonraki deney numarasını al ve sayacı artır"""
        cls._current_experiment_num += 1
        return cls._current_experiment_num
    
    @classmethod
    def reset_experiment_counter(cls):
        """Deney sayacını ve en iyi deney bilgilerini sıfırla"""
        cls._current_experiment_num = 0
        cls._best_experiment = {
            'name': None,
            'accuracy': 0.0,
            'config': None,
            'results': None
        }
    
    def _create_model(self, input_size: int, activation: str) -> Model:
        """CNN modelini oluştur"""
        inputs = layers.Input(shape=(input_size, input_size, 3))
        
        def apply_activation(x, activation_type):
            if activation_type == 'prelu':
                return layers.PReLU()(x)
            return layers.Activation(activation_type)(x)
        
        # L2 regularizer'ı oluştur
        regularizer = tf.keras.regularizers.l2(self.config.model.weight_decay)
        
        # İlk Konvolüsyon Bloğu
        x = layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizer)(inputs)
        x = apply_activation(x, activation)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        
        # İkinci Konvolüsyon Bloğu
        x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizer)(x)
        x = apply_activation(x, activation)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        
        # Üçüncü Konvolüsyon Bloğu
        x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizer)(x)
        x = apply_activation(x, activation)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        
        # Fully Connected Katmanları
        x = layers.Flatten()(x)
        x = layers.Dense(256, kernel_regularizer=regularizer)(x)
        x = apply_activation(x, activation)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config.model.dropout_rate)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return Model(inputs, outputs)
    
    def _create_callbacks(self, experiment_dir: str) -> list:
        """Model callback'lerini oluştur"""

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(experiment_dir, 'best_model.keras'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            wandb.keras.WandbMetricsLogger(log_freq='epoch')
        ]
        return callbacks
    
    def _print_test_results(self, test_results: Dict[str, list], experiment_name: str) -> None:
        """Test sonuçlarını yazdır"""
        print(f"\n{experiment_name} Test Sonuçları:")
        print("-" * 50)
        
        # Orijinal test sonuçları
        print("Orijinal Test:")
        print(f"Loss: {test_results['original'][0]:.4f}")
        print(f"Accuracy: {test_results['original'][1]:.4f}")
        
        # Manipüle edilmiş test sonuçları
        print("\nManipüle Edilmiş Test:")
        print(f"Loss: {test_results['manipulated'][0]:.4f}")
        print(f"Accuracy: {test_results['manipulated'][1]:.4f}")
        
        # Renk sabitliği uygulanmış test sonuçları
        print("\nRenk Sabitliği Test:")
        print(f"Loss: {test_results['white_balanced'][0]:.4f}")
        print(f"Accuracy: {test_results['white_balanced'][1]:.4f}")
        print("-" * 50)
        
        # En iyi deneyi güncelle
        avg_accuracy = (test_results['original'][1] + test_results['manipulated'][1] + test_results['white_balanced'][1]) / 3
        if avg_accuracy > self._best_experiment['accuracy']:
            self._best_experiment['name'] = experiment_name
            self._best_experiment['accuracy'] = avg_accuracy
            self._best_experiment['config'] = self.current_experiment
            self._best_experiment['results'] = test_results
    
    @classmethod
    def print_best_experiment(cls):
        """En başarılı deneyin sonuçlarını yazdır"""
        if cls._best_experiment['name'] is None:
            print("\nHenüz hiç deney yapılmamış!")
            return
            
        print("\n" + "="*20 + " EN BAŞARILI DENEY " + "="*20)
        print(f"Deney Adı: {cls._best_experiment['name']}")
        print("\nKonfigürasyon:")
        print(f"Input Size: {cls._best_experiment['config']['input_size']}")
        print(f"Aktivasyon: {cls._best_experiment['config']['activation']}")
        print(f"Learning Rate: {cls._best_experiment['config']['learning_rate']}")
        
        print("\nTest Sonuçları:")
        print("Orijinal Test Accuracy: {:.4f}".format(cls._best_experiment['results']['original'][1]))
        print("Manipüle Test Accuracy: {:.4f}".format(cls._best_experiment['results']['manipulated'][1]))
        print("Renk Sabitliği Test Accuracy: {:.4f}".format(cls._best_experiment['results']['white_balanced'][1]))
        print("Ortalama Accuracy: {:.4f}".format(cls._best_experiment['accuracy']))
        print("="*60)
    
    def _create_data_generator(self, use_augmentation: bool) -> ImageDataGenerator:
        """Data generator oluştur"""
        if use_augmentation:
            params = self.config.model.data_augmentation.params
            return ImageDataGenerator(
                rotation_range=params.rotation_range,
                width_shift_range=params.width_shift_range,
                height_shift_range=params.height_shift_range,
                horizontal_flip=params.horizontal_flip,
                zoom_range=params.zoom_range,
                brightness_range=params.brightness_range,
                shear_range=params.shear_range,
                channel_shift_range=params.channel_shift_range,
                rescale=1./255
            )
        else:
            return ImageDataGenerator(rescale=1./255)
    
    def _dataset_to_numpy(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """TensorFlow dataset'i numpy array'e dönüştür"""
        images = []
        labels = []
        for image_batch, label_batch in dataset:
            images.append(image_batch.numpy())
            labels.append(label_batch.numpy())
        return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)
    
    def train_and_evaluate(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        test_manip_ds: tf.data.Dataset,
        test_wb_ds: tf.data.Dataset,
        input_size: int
    ) -> None:
        """Grid Search ile model eğitimi ve değerlendirmesi"""
        
        # Dataset'leri numpy array'e dönüştür
        print("\nDataset'ler numpy array'e dönüştürülüyor...")
        train_images, train_labels = self._dataset_to_numpy(train_ds)
        val_images, val_labels = self._dataset_to_numpy(val_ds)
        test_images, test_labels = self._dataset_to_numpy(test_ds)
        test_manip_images, test_manip_labels = self._dataset_to_numpy(test_manip_ds)
        test_wb_images, test_wb_labels = self._dataset_to_numpy(test_wb_ds)
        
        # Veri setlerinin kontrolü
        print("\nVeri seti kontrolleri:")
        print(f"Train veri seti: {len(train_images)} örnek")
        print(f"Validation veri seti: {len(val_images)} örnek")
        print(f"Test veri seti: {len(test_images)} örnek")
        print(f"Test manipüle veri seti: {len(test_manip_images)} örnek")
        print(f"Test renk sabitliği veri seti: {len(test_wb_images)} örnek")
        
        for activation in self.config.model.activations:
            for lr in self.config.model.learning_rates:
                for use_augmentation in self.config.model.data_augmentation.enabled:
                    current_experiment_num = self.get_next_experiment_num()
                    
                    # Experiment konfigürasyonu
                    self.current_experiment = {
                        'input_size': input_size,
                        'activation': activation,
                        'learning_rate': lr,
                        'data_augmentation': use_augmentation
                    }
                    
                    # WandB run başlat
                    experiment_name = f"exp_size_{input_size}_act_{activation}_lr_{lr}_aug_{use_augmentation}"
                    print(f"\n{'='*20} Deney {current_experiment_num}/{self.total_experiments} {'='*20}")
                    print(f"Deney Adı: {experiment_name}")
                    print(f"Input Size: {input_size}")
                    print(f"Activation: {activation}")
                    print(f"Learning Rate: {lr}")
                    print(f"Data Augmentation: {'Aktif' if use_augmentation else 'Pasif'}")
                    
                    # Wandb dizinini result altında oluştur
                    wandb_dir = Path(self.config.paths.results) / "wandb"
                    wandb_dir.mkdir(parents=True, exist_ok=True)
                    
                    wandb.init(
                        project=self.config.wandb.project,
                        entity=self.config.wandb.entity,
                        name=experiment_name,
                        dir=str(wandb_dir),
                        config={
                            **self.current_experiment,
                            'batch_size': self.config.model.batch_size,
                            'loss_function': 'sparse_categorical_crossentropy',
                            'kernel_size': 3,  # CNN katmanlarındaki kernel boyutu
                            'filters': [32, 64, 128],  # CNN katmanlarındaki filtre sayıları
                            'optimizer': 'Adam',
                            'dropout_rate': self.config.model.dropout_rate,
                            'weight_decay': self.config.model.weight_decay
                        }
                    )
                    
                    # Experiment dizini oluştur
                    experiment_dir = Path(self.config.paths.checkpoints) / experiment_name
                    experiment_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Her deney için yeni bir model oluştur
                    model = self._create_model(input_size, activation)
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Model mimarisini yazdır
                    print("\nModel Mimarisi:")
                    print("-" * 50)
                    model.summary()
                    print("-" * 50)
                    
                    # Data generator oluştur
                    data_generator = self._create_data_generator(use_augmentation)
                    
                    # Model eğitimi
                    print(f"\nEğitim başlıyor...")
                    if use_augmentation:
                        history = model.fit(
                            data_generator.flow(train_images, train_labels, batch_size=self.config.model.batch_size),
                            validation_data=(val_images, val_labels),
                            epochs=self.config.model.epochs,
                            callbacks=self._create_callbacks(str(experiment_dir)),
                            verbose=1
                        )
                    else:
                        history = model.fit(
                            train_images,
                            train_labels,
                            validation_data=(val_images, val_labels),
                            batch_size=self.config.model.batch_size,
                            epochs=self.config.model.epochs,
                            callbacks=self._create_callbacks(str(experiment_dir)),
                            verbose=1
                        )
                    
                    # Eğitim geçmişini yazdır
                    print("\nEğitim geçmişi:")
                    for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                        history.history['loss'],
                        history.history['accuracy'],
                        history.history['val_loss'],
                        history.history['val_accuracy']
                    )):
                        print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                    
                    # Test değerlendirmesi
                    print("\nTest değerlendirmesi yapılıyor...")
                    test_results = {
                        'original': model.evaluate(test_images, test_labels, verbose=0),
                        'manipulated': model.evaluate(test_manip_images, test_manip_labels, verbose=0),
                        'white_balanced': model.evaluate(test_wb_images, test_wb_labels, verbose=0)
                    }
                    
                    # Test sonuçlarını yazdır
                    self._print_test_results(test_results, experiment_name)
                    
                    # Sonuçları WandB'ye kaydet
                    wandb.log({
                        'test_original_loss': test_results['original'][0],
                        'test_original_accuracy': test_results['original'][1],
                        'test_manipulated_loss': test_results['manipulated'][0],
                        'test_manipulated_accuracy': test_results['manipulated'][1],
                        'test_wb_loss': test_results['white_balanced'][0],
                        'test_wb_accuracy': test_results['white_balanced'][1]
                    })
                    
                    wandb.finish()
                    
                    print(f"\nDeney {current_experiment_num}/{self.total_experiments} tamamlandı.")
                    
                    # Belleği temizle
                    tf.keras.backend.clear_session()