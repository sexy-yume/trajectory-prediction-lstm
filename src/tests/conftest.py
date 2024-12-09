import pytest
import torch
import json
from pathlib import Path
from src.config.config import Config

@pytest.fixture
def config():
    config_path = Path(__file__).parent / "fixtures" / "test_config.yaml"
    return Config.from_yaml(str(config_path))

@pytest.fixture
def sample_sequence():
    with open(Path(__file__).parent / "fixtures" / "sample_sequence.json", "r") as f:
        return json.load(f)

@pytest.fixture
def sample_features():
    return torch.randn(32, 10, 12)  # batch_size, seq_length, feature_dim

@pytest.fixture
def sample_targets():
    return torch.randn(32, 2)  # batch_size, output_dim
