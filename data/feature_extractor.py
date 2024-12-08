import numpy as np
import torch
import logging
from pathlib import Path
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence

from ..config.config import Config

class TrajectoryFeatureExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.window_size = config.features.window_size
        self.cache_dir = config.paths.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_path(self, json_path: str) -> Path:
        file_stat = Path(json_path).stat()
        hash_string = f"{json_path}_{file_stat.st_mtime}_{file_stat.st_size}_w{self.window_size}"
        cache_hash = hashlib.md5(hash_string.encode()).hexdigest()
        return self.cache_dir / f"lstm_dataset_cache_{cache_hash}.pt"

    def _load_from_cache(self, cache_path: Path) -> Optional[Tuple[list, list]]:
        try:
            self.logger.info(f"Loading cached dataset... ({cache_path})")
            cached_data = torch.load(cache_path)
            return cached_data['features'], cached_data['targets']
        except (EOFError, FileNotFoundError, RuntimeError) as e:
            self.logger.warning(f"Cache load failed: {str(e)}")
            return None

    def _save_to_cache(self, cache_path: Path, features: list, targets: list):
        self.logger.info(f"Caching dataset... ({cache_path})")
        cache_data = {
            'features': features,
            'targets': targets,
            'window_size': self.window_size,
            'cache_version': '1.0'
        }
        torch.save(cache_data, cache_path)

    def compute_velocity_changes(self, velocities: np.ndarray) -> np.ndarray:
        velocity_diff = np.diff(velocities, axis=0)
        return np.concatenate([[0], velocity_diff])
    
    def extract_frame_features(self, frame: Dict, prev_frame: Optional[Dict] = None) -> np.ndarray:
        flows = frame['flows']
        if not flows:
            return np.zeros(12)
            
        positions = np.array([[f['start_x'], f['start_y']] for f in flows])
        velocities = np.array([f['velocity'] for f in flows])
        directions = np.array([f['direction'] for f in flows])
        
        center = positions.mean(axis=0)
        
        features = []
        
        features.extend([
            velocities.mean(),
            velocities.std(),
            np.max(velocities),
            np.min(velocities)
        ])
        
        features.extend([
            directions.mean(),
            directions.std()
        ])
        
        if prev_frame is not None and prev_frame['flows']:
            prev_velocities = np.array([f['velocity'] for f in prev_frame['flows']])
            velocity_change = velocities.mean() - prev_velocities.mean()
            prev_directions = np.array([f['direction'] for f in prev_frame['flows']])
            direction_change = directions.mean() - prev_directions.mean()
        else:
            velocity_change = direction_change = 0
            
        features.extend([velocity_change, direction_change])
        
        dist_from_center = np.linalg.norm(positions - center, axis=1)
        features.extend([
            dist_from_center.mean(),
            dist_from_center.std()
        ])
        
        return np.array(features)
    
    def extract_sequence_features(self, sequence: Dict) -> np.ndarray:
        frames = sequence['frames']
        n_frames = len(frames)
        
        frame_features = []
        for i in range(n_frames):
            prev_frame = frames[i-1] if i > 0 else None
            features = self.extract_frame_features(frames[i], prev_frame)
            frame_features.append(features)
            
        frame_features = np.array(frame_features)
        
        windowed_features = []
        for i in range(n_frames):
            window_start = max(0, i - self.window_size)
            window_data = frame_features[window_start:i+1]
            
            if len(window_data) > 0:
                window_stats = np.concatenate([
                    window_data.mean(axis=0),
                    window_data.std(axis=0)
                ])
            else:
                window_stats = np.zeros(frame_features.shape[1] * 2)
                
            current_features = frame_features[i]
            windowed_features.append(np.concatenate([
                current_features,
                window_stats
            ]))
            
        return np.array(windowed_features)
    
    def prepare_dataset(self) -> Tuple[list, list]:
        json_path = self.config.paths.data_path
        cache_path = self._get_cache_path(json_path)
        
        if self.config.features.cache_enabled:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                self.logger.info("Successfully loaded cached dataset.")
                return cached_data

        self.logger.info("Preparing new dataset...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        features = []
        targets = []
        
        sequences = data['sequences'] if 'sequences' in data else [data]
        
        for sequence in sequences:
            if sequence['metadata']['normalized_ground_truth'] is None:
                continue
                
            sequence_features = self.extract_sequence_features(sequence)
            features.append(torch.FloatTensor(sequence_features))
            targets.append(torch.FloatTensor(
                sequence['metadata']['normalized_ground_truth']
            ))

        if self.config.features.cache_enabled:
            self._save_to_cache(cache_path, features, targets)
        
        return features, targets
