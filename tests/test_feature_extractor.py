import pytest
import torch
import numpy as np
from src.data.feature_extractor import TrajectoryFeatureExtractor

def test_feature_extractor_initialization(config):
    extractor = TrajectoryFeatureExtractor(config)
    assert extractor.window_size == config.features.window_size
    assert extractor.cache_dir == config.paths.cache_dir

def test_extract_frame_features(config, sample_sequence):
    extractor = TrajectoryFeatureExtractor(config)
    features = extractor.extract_frame_features(sample_sequence['frames'][0])
    assert isinstance(features, np.ndarray)
    assert features.shape == (12,)
