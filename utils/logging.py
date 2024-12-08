import logging
from datetime import datetime
import wandb
from pathlib import Path
import sys
from typing import Optional

from ..config.config import Config

def setup_logging(config: Config) -> None:
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = getattr(logging, config.logging.log_level.upper())
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if config.logging.log_file:
        log_file = Path(config.logging.log_file)
        if not log_file.parent.exists():
            log_file.parent.mkdir(parents=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

def init_wandb(config: Config) -> Optional[wandb.Run]:
    if not config.logging.wandb_enabled:
        return None
    
    try:
        wandb.login()
        return wandb.init(
            project=config.logging.project_name,
            config=config.__dict__,
            dir=str(config.paths.model_dir)
        )
    except Exception as e:
        logging.warning(f"Failed to initialize wandb: {e}")
        return None

def log_hyperparameters(config: Config) -> None:
    logging.info("Configuration:")
    logging.info(f"Model:")
    logging.info(f"  Input size: {config.model.input_size}")
    logging.info(f"  Hidden size: {config.model.hidden_size}")
    logging.info(f"  Num layers: {config.model.num_layers}")
    logging.info(f"  Dropout: {config.model.dropout}")
    
    logging.info(f"Training:")
    logging.info(f"  Batch size: {config.training.batch_size}")
    logging.info(f"  Learning rate: {config.training.learning_rate}")
    logging.info(f"  Weight decay: {config.training.weight_decay}")
    logging.info(f"  Epochs: {config.training.num_epochs}")
    
    logging.info(f"Hardware:")
    logging.info(f"  Device: {config.hardware.device}")
    logging.info(f"  Num workers: {config.hardware.num_workers}")

class MetricLogger:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.wandb_enabled = config.logging.wandb_enabled
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        metric_str = " - ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metric_str}" if step is not None else metric_str)
        
        if self.wandb_enabled:
            if step is not None:
                metrics['step'] = step
            wandb.log(metrics)
    
    def log_model_predictions(self, predictions: dict) -> None:
        if not self.wandb_enabled:
            return
            
        try:
            wandb.log(predictions)
        except Exception as e:
            self.logger.warning(f"Failed to log predictions to wandb: {e}")

def log_exception(logger: logging.Logger, exc: Exception) -> None:
    logger.error(f"Exception occurred: {exc.__class__.__name__}", exc_info=True)

class WandbCallback:
    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.logging.wandb_enabled
    
    def on_train_begin(self) -> None:
        if not self.enabled:
            return
        init_wandb(self.config)
    
    def on_train_end(self) -> None:
        if not self.enabled:
            return
        wandb.finish()
    
    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        if not self.enabled:
            return
        metrics['epoch'] = epoch
        wandb.log(metrics)
