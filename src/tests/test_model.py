import pytest
import torch
from src.models.lstm import AttentionLSTM

def test_model_initialization(config):
    model = AttentionLSTM(config)
    assert isinstance(model, torch.nn.Module)

def test_model_forward(config, sample_features):
    model = AttentionLSTM(config)
    output = model(sample_features)
    assert output.shape == (32, 2)  # batch_size, output_dim
