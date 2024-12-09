import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import argparse
from typing import Tuple, List

from src.config.config import Config
from src.data.feature_extractor import TrajectoryFeatureExtractor
from src.models.lstm import AttentionLSTM
from src.training.trainer import TrajectoryTrainer
from src.utils.logging import setup_logging, log_hyperparameters, log_exception

def parse_args():
    parser = argparse.ArgumentParser(description='Train trajectory prediction model')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    return parser.parse_args()

def prepare_dataloaders(config: Config, features: List[torch.Tensor], 
                       targets: List[torch.Tensor]) -> Tuple[DataLoader, DataLoader]:
    dataset_size = len(features)
    val_size = int(0.15 * dataset_size)
    train_size = dataset_size - val_size
    
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    val_features = features[train_size:]
    val_targets = targets[train_size:]
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.stack([f.to(torch.float32) for f in train_features]),
        torch.stack([t.to(torch.float32) for t in train_targets])
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.stack([f.to(torch.float32) for f in val_features]),
        torch.stack([t.to(torch.float32) for t in val_targets])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    args = parse_args()
    
    try:
        config = Config.from_yaml(args.config)
        setup_logging(config)
        logger = logging.getLogger(__name__)
        logger.info("Starting training process...")
        
        log_hyperparameters(config)
        
        logger.info("Initializing feature extractor...")
        feature_extractor = TrajectoryFeatureExtractor(config)
        
        logger.info("Loading and preprocessing data...")
        features, targets = feature_extractor.prepare_dataset() 
        
        config.model.input_size = features[0].shape[1]
        logger.info(f"Input size set to {config.model.input_size}")
        
        logger.info("Preparing dataloaders...")
        train_loader, val_loader = prepare_dataloaders(config, features, targets)
        
        logger.info("Initializing trainer...")
        trainer = TrajectoryTrainer(config)
        
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        logger.info("Starting training...")
        trainer.train(train_loader, val_loader)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        log_exception(logger, e)
        raise

if __name__ == "__main__":
    main()
