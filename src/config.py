from dataclasses import dataclass
from typing import List, Dict, Any
import yaml
from pathlib import Path


@dataclass
class DatasetConfig:
    base_path: str
    train_split: float
    val_split: float
    save_path: str
    classes: List[str]


@dataclass
class DataAugmentationParams:
    rotation_range: int
    width_shift_range: float
    height_shift_range: float
    horizontal_flip: bool
    zoom_range: float
    brightness_range: List[float]
    shear_range: int
    channel_shift_range: float


@dataclass
class DataAugmentationConfig:
    enabled: List[bool]
    params: DataAugmentationParams


@dataclass
class ModelConfig:
    input_sizes: List[int]
    activations: List[str]
    learning_rates: List[float]
    batch_size: int
    epochs: int
    dropout_rate: float
    weight_decay: float
    data_augmentation: DataAugmentationConfig


@dataclass
class WandbConfig:
    project: str
    entity: str
    api_key: str


@dataclass
class PathsConfig:
    checkpoints: str
    logs: str
    results: str


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    wandb: WandbConfig
    paths: PathsConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """YAML dosyasından config yükleme"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Data augmentation parametrelerini yapılandır
        aug_params = config_dict['model']['data_augmentation']['params']
        data_aug_config = DataAugmentationConfig(
            enabled=config_dict['model']['data_augmentation']['enabled'],
            params=DataAugmentationParams(**aug_params)
        )
        config_dict['model']['data_augmentation'] = data_aug_config

        return cls(
            dataset=DatasetConfig(**config_dict['dataset']),
            model=ModelConfig(**config_dict['model']),
            wandb=WandbConfig(**config_dict['wandb']),
            paths=PathsConfig(**config_dict['paths'])
        )

    def create_directories(self) -> None:
        """Gerekli klasörleri oluşturma"""
        directories = [
            self.dataset.save_path,
            self.paths.checkpoints,
            self.paths.logs,
            self.paths.results
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True) 