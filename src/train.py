import tensorflow as tf
import os
from pathlib import Path
from typing import Tuple, Dict
from config import Config
from dataset import Dataset
from model import CNNModel

# Tensorflow uyarılarını sustur
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def create_dataset(
    data_dir: str,
    input_size: int,
    batch_size: int
) -> tf.data.Dataset:
    """TensorFlow dataset oluştur ve normalize et"""
    dataset_type = os.path.basename(data_dir)
    print("-"*20 + f"{dataset_type.upper()} Veri Seti Eğitim için Hazırlanıyor. ## input_size: {input_size}" + "-"*20)
            
    # Normal dataset oluşturma işlemine devam et
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(input_size, input_size),
        batch_size=batch_size,
        shuffle=True
    )
    
    # Normalizasyon işlemi - piksel değerlerini [0,1] aralığına getir
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    
    return normalized_dataset


def main():
    # Config yükle
    config = Config.from_yaml('config/config.yaml')
    config.create_directories()
    
    # WandB API key'i ayarla
    os.environ["WANDB_API_KEY"] = config.wandb.api_key
    
    print("-"*20+"Dataset Bilgileri"+"-"*20)
    print("Sınıf sayısı:", len(config.dataset.classes))
    print("Sınıflar:", config.dataset.classes)
    
    # Dataset hazırla
    dataset = Dataset(config)
    dataset.prepare_dataset()
    
    # Deney sayacını sıfırla
    CNNModel.reset_experiment_counter()
    
    # Her input size için ayrı model ve dataset oluştur
    for input_size in config.model.input_sizes:
        print(f"\nInput size {input_size} için dataset hazırlanıyor...")
        
        # Bu input size için dataset'leri oluştur
        current_datasets = {
            'train': create_dataset(
                os.path.join(config.dataset.save_path, 'train'),
                input_size,
                config.model.batch_size
            ),
            'val': create_dataset(
                os.path.join(config.dataset.save_path, 'val'),
                input_size,
                config.model.batch_size
            ),
            'test': create_dataset(
                os.path.join(config.dataset.save_path, 'test'),
                input_size,
                config.model.batch_size
            ),
            'test_manipulated': create_dataset(
                os.path.join(config.dataset.save_path, 'test_manipulated'),
                input_size,
                config.model.batch_size
            ),
            'test_wb': create_dataset(
                os.path.join(config.dataset.save_path, 'test_wb'),
                input_size,
                config.model.batch_size
            )
        }
        
        # Dataset'leri önbellekle ve prefetch yap
        for ds_name, ds in current_datasets.items():
            current_datasets[ds_name] = ds.cache().prefetch(tf.data.AUTOTUNE)
        
        # Bu input size için yeni bir model oluştur
        num_classes = len(config.dataset.classes)
        model = CNNModel(config, num_classes)
        
        # Model eğitimi
        model.train_and_evaluate(
            current_datasets['train'],
            current_datasets['val'],
            current_datasets['test'],
            current_datasets['test_manipulated'],
            current_datasets['test_wb'],
            input_size
        )

    # Tüm deneyler tamamlandıktan sonra en başarılı deneyi yazdır
    print("\nTüm deneyler tamamlandı!")
    CNNModel.print_best_experiment()


if __name__ == '__main__':
    main() 