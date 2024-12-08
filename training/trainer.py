import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple

from ..config.config import Config
from ..models.lstm import AttentionLSTM

class TrajectoryTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.hardware.device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self.logger = logging.getLogger(__name__)
        
    def initialize_model(self):
        self.model = AttentionLSTM(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            features, targets = batch
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            lengths = torch.sum(torch.any(features != 0, dim=2), dim=1)
            
            self.optimizer.zero_grad()
            predictions = self.model(features, lengths)
            loss = nn.MSELoss()(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                features, targets = batch
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                lengths = torch.sum(torch.any(features != 0, dim=2), dim=1)
                predictions = self.model(features, lengths)
                loss = nn.MSELoss()(predictions, targets)
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        if self.config.logging.wandb_enabled:
            wandb.init(project=self.config.logging.project_name, config=self.config.__dict__)
        
        self.initialize_model()
        
        for epoch in range(self.config.training.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            if self.config.logging.wandb_enabled:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch': epoch,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            self.logger.info(
                f"Epoch {epoch}: Train Loss = {train_loss:.6f}, "
                f"Val Loss = {val_loss:.6f}"
            )
            
            if val_loss < self.best_val_loss - self.config.training.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.training.patience:
                self.logger.info("Early stopping triggered")
                break
                
        if self.config.logging.wandb_enabled:
            wandb.finish()
        
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, Path(self.config.paths.model_dir) / filename)
        
    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(Path(self.config.paths.model_dir) / filename)
        self.config = checkpoint['config']
        self.initialize_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
