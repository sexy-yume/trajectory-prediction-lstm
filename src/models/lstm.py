import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple

from ..config.config import Config

class AttentionLSTM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            batch_first=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(config.model.hidden_size, config.model.hidden_size),
            nn.Tanh(),
            nn.Linear(config.model.hidden_size, 1)
        )
        
        self.fc = nn.Linear(config.model.hidden_size, 2)
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed_x)
            outputs, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            outputs, _ = self.lstm(x)
            
        attention_weights = F.softmax(self.attention(outputs), dim=1)
        context = torch.sum(outputs * attention_weights, dim=1)
        predictions = self.fc(context)
        return predictions

class TrajectoryPredictor:
    def __init__(self, model_path: str, config: Config):
        self.device = torch.device('cpu')
        self.config = config
        self.model = AttentionLSTM(config).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        features = features.unsqueeze(0).to(self.device)
        predictions = self.model(features)
        return predictions.cpu()
    
    @torch.no_grad()
    def predict_batch(self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = features.to(self.device)
        predictions = self.model(features, lengths)
        return predictions.cpu()
