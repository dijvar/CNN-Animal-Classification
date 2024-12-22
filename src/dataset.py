import os
from typing import Tuple, List, Dict
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import Config
from utils import (
    manipulate_light_source,
    process_and_white_balance,
    get_light_sources
)


class Dataset:
    def __init__(self, config: Config):
        self.config = config
        self.class_names: List[str] = self.config.dataset.classes
        self.class_to_idx: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
    def _load_image(self, image_path: str) -> np.ndarray:
        """Görüntüyü yükle ve normalize et"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img / 255.0
    
    def _save_image(self, image: np.ndarray, save_path: str) -> None:
        """Görüntüyü kaydet ve denormalize et"""
        # Değerleri 0-1 aralığında sınırla
        image = np.clip(image, 0, 1)
        # 0-255 aralığına dönüştür
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)
    
    def _create_manipulated_images(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Manipüle edilmiş görüntüleri oluştur"""
        purplish_light, yellowish_light, greenish_light = get_light_sources()
        
        manipulated_purplish = manipulate_light_source(image, purplish_light)
        manipulated_yellowish = manipulate_light_source(image, yellowish_light)
        manipulated_greenish = manipulate_light_source(image, greenish_light)
        
        return manipulated_purplish, manipulated_yellowish, manipulated_greenish
    
    def _apply_wb(self, image: np.ndarray) -> np.ndarray:
        """Renk sabitliği uygula"""
        return process_and_white_balance(image)
    
    def _check_dataset_exists(self) -> bool:
        """İşlenmiş veri setinin varlığını ve geçerliliğini kontrol et"""
        if not os.path.exists(self.config.dataset.save_path):
            return False
            
        # Tüm gerekli klasörleri kontrol et
        required_splits = ['train', 'val', 'test', 'test_manipulated', 'test_wb']
        for split in required_splits:
            split_path = os.path.join(self.config.dataset.save_path, split)
            if not os.path.exists(split_path):
                print(f"Eksik klasör: {split_path}")
                return False
            
            # Her sınıf için klasör ve görüntü kontrolü
            for class_name in self.class_names:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    print(f"Eksik sınıf klasörü: {class_path}")
                    return False
                
                # Klasörde görüntü var mı kontrol et
                images = list(Path(class_path).glob('*.jpg')) + list(Path(class_path).glob('*.png'))
                if not images:
                    print(f"Görüntü bulunamadı: {class_path}")
                    return False
                
                # Rastgele bir görüntüyü açmayı dene
                try:
                    test_img = cv2.imread(str(images[0]))
                    if test_img is None:
                        print(f"Bozuk görüntü tespit edildi: {images[0]}")
                        return False
                except Exception as e:
                    print(f"Görüntü okuma hatası: {str(e)}")
                    return False
        
        print("\n\nVeri seti kontrolü başarılı: Tüm klasörler ve görüntüler mevcut.")
        return True
    
    def prepare_dataset(self) -> None:
        """Veri setini hazırla"""
        # Veri seti kontrolü
        if self._check_dataset_exists():
            print("\nVeri seti zaten hazırlanmış ve geçerli.")
            return
        else:
            print("\nVeri seti hazırlanıyor...")
        
        # Her sınıf için klasörleri oluştur
        for split in ['train', 'val', 'test', 'test_manipulated', 'test_wb']:
            for class_name in self.class_names:
                os.makedirs(os.path.join(self.config.dataset.save_path, split, class_name), exist_ok=True)
        
        # Her sınıf için
        for class_name in self.class_names:
            print(f"\nSınıf işleniyor: {class_name}")
            class_path = Path(self.config.dataset.base_path) / class_name
            
            if not class_path.exists():
                print(f"Uyarı: {class_name} için klasör bulunamadı: {class_path}")
                continue
                
            image_paths = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            
            if not image_paths:
                print(f"Uyarı: {class_name} için görüntü bulunamadı")
                continue
            
            print(f"Toplam görüntü sayısı: {len(image_paths)}")
            
            # Train-test split
            total_images = len(image_paths)
            test_size = int(total_images * (1 - self.config.dataset.train_split))
            train_paths, test_paths = train_test_split(
                image_paths,
                test_size=test_size,
                random_state=42
            )
            
            # Train-validation split
            val_size = int(len(train_paths) * self.config.dataset.val_split)
            train_paths, val_paths = train_test_split(
                train_paths,
                test_size=val_size,
                random_state=42
            )
            
            print(f"Train: {len(train_paths)}, Validation: {len(val_paths)}, Test: {len(test_paths)}")
            
            # Train ve validation görüntülerini kaydet
            for paths, split in [(train_paths, 'train'), (val_paths, 'val')]:
                for img_path in paths:
                    try:
                        img = self._load_image(str(img_path))
                        save_path = os.path.join(self.config.dataset.save_path, split, class_name, img_path.name)
                        self._save_image(img, save_path)
                    except Exception as e:
                        print(f"Hata: {img_path} işlenirken hata oluştu: {str(e)}")
            
            # Test görüntülerini işle ve kaydet
            for img_path in test_paths:
                try:
                    img = self._load_image(str(img_path))
                    
                    # Orijinal test görüntüsünü kaydet
                    test_save_path = os.path.join(self.config.dataset.save_path, 'test', class_name, img_path.name)
                    self._save_image(img, test_save_path)
                    
                    # Manipüle edilmiş görüntüleri kaydet
                    manip_images = self._create_manipulated_images(img)
                    for idx, manip_img in enumerate(manip_images):
                        manip_save_path = os.path.join(
                            self.config.dataset.save_path,
                            'test_manipulated',
                            class_name,
                            f"{img_path.stem}_manip_{idx}{img_path.suffix}"
                        )
                        self._save_image(manip_img, manip_save_path)
                        
                        # Renk sabitliği uygula ve kaydet
                        wb_img = self._apply_wb(manip_img)
                        wb_save_path = os.path.join(
                            self.config.dataset.save_path,
                            'test_wb',
                            class_name,
                            f"{img_path.stem}_wb_{idx}{img_path.suffix}"
                        )
                        self._save_image(wb_img, wb_save_path)
                except Exception as e:
                    print(f"Hata: {img_path} işlenirken hata oluştu: {str(e)}")
        
        print("\nVeri seti hazırlama tamamlandı.") 