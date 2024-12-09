from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Optional
import logging

@dataclass
class ModelConfig:
   input_size: int
   hidden_size: int
   num_layers: int
   dropout: float

   def __post_init__(self):
       if self.input_size <= 0:
           raise ValueError("input_size must be positive")
       if self.hidden_size <= 0:
           raise ValueError("hidden_size must be positive")
       if self.num_layers <= 0:
           raise ValueError("num_layers must be positive")
       if not 0 <= self.dropout < 1:
           raise ValueError("dropout must be between 0 and 1")

@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    min_delta: float

    def __post_init__(self):
        # 문자열을 숫자로 변환
        self.batch_size = int(self.batch_size)
        self.num_epochs = int(self.num_epochs)
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.patience = int(self.patience)
        self.min_delta = float(self.min_delta)

        # 값 검증
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.min_delta <= 0:
            raise ValueError("min_delta must be positive")

@dataclass
class HardwareConfig:
   device: str
   num_workers: int

   def __post_init__(self):
       if self.device not in ['cuda', 'cpu']:
           raise ValueError("device must be either 'cuda' or 'cpu'")
       if self.num_workers < 0:
           raise ValueError("num_workers must be non-negative")

@dataclass
class PathConfig:
   model_dir: str
   cache_dir: str
   data_dir: str
   data_file: str
   data_path: str

   def __post_init__(self):
       self.model_dir = Path(self.model_dir)
       self.cache_dir = Path(self.cache_dir)
       self.data_dir = Path(self.data_dir)
       self.data_path = Path(self.data_path)
       
       self.model_dir.mkdir(parents=True, exist_ok=True)
       self.cache_dir.mkdir(parents=True, exist_ok=True)
       self.data_dir.mkdir(parents=True, exist_ok=True)
       
       if not self.data_path.exists():
           raise FileNotFoundError(f"Data file not found: {self.data_path}")
       
       if not str(self.data_path).startswith(str(self.data_dir)):
           raise ValueError(f"Data file must be in data_dir: {self.data_dir}")

@dataclass
class LoggingConfig:
   project_name: str
   wandb_enabled: bool
   log_level: str
   log_file: str

   def __post_init__(self):
       valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
       if self.log_level not in valid_levels:
           raise ValueError(f"log_level must be one of {valid_levels}")

@dataclass
class FeatureConfig:
   window_size: int
   cache_enabled: bool

   def __post_init__(self):
       if self.window_size <= 0:
           raise ValueError("window_size must be positive")

@dataclass
class Config:
   model: ModelConfig
   training: TrainingConfig
   hardware: HardwareConfig
   paths: PathConfig
   logging: LoggingConfig
   features: FeatureConfig

   @classmethod
   def from_yaml(cls, yaml_path: str) -> 'Config':
       try:
           with open(yaml_path, 'r') as f:
               config_dict = yaml.safe_load(f)
       except FileNotFoundError:
           raise FileNotFoundError(f"Config file not found: {yaml_path}")
       except yaml.YAMLError as e:
           raise ValueError(f"Error parsing config file: {e}")

       try:
           return cls(
               model=ModelConfig(**config_dict['model']),
               training=TrainingConfig(**config_dict['training']),
               hardware=HardwareConfig(**config_dict['hardware']),
               paths=PathConfig(**config_dict['paths']),
               logging=LoggingConfig(**config_dict['logging']),
               features=FeatureConfig(**config_dict['features'])
           )
       except KeyError as e:
           raise KeyError(f"Missing configuration key: {e}")
       except TypeError as e:
           raise TypeError(f"Invalid configuration value: {e}")

   def save(self, yaml_path: str):
       config_dict = {
           'model': self.model.__dict__,
           'training': self.training.__dict__,
           'hardware': self.hardware.__dict__,
           'paths': {
               'model_dir': str(self.paths.model_dir),
               'cache_dir': str(self.paths.cache_dir),
               'data_dir': str(self.paths.data_dir),
               'data_file': str(self.paths.data_file),
               'data_path': str(self.paths.data_path)
           },
           'logging': self.logging.__dict__,
           'features': self.features.__dict__
       }

       try:
           with open(yaml_path, 'w') as f:
               yaml.dump(config_dict, f, default_flow_style=False)
       except Exception as e:
           raise IOError(f"Error saving config file: {e}")

   def __post_init__(self):
       if self.model.input_size != self.features.window_size * 12:
           logging.warning(
               f"Input size ({self.model.input_size}) might not match "
               f"feature dimension ({self.features.window_size * 12})"
           )
