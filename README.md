
# Trajectory Prediction LSTM

A PyTorch-based implementation of trajectory prediction using LSTM with attention mechanism. This package provides tools for training and deploying models that predict object trajectories using deep learning.
![Test Result](https://github.com/sexy-yume/trajectory-prediction-lstm/raw/main/test.png)
---

## Features

- LSTM with attention mechanism
- GPU acceleration support
- Flexible feature extraction
- Configurable training pipeline
- Wandb integration for experiment tracking
- Dataset caching for faster training
- CPU optimized inference

---

## Installation

### Using pip
```bash
pip install trajectory-prediction-lstm
```

### From source
```bash
git clone https://github.com/sexy-yume/trajectory-prediction.git
cd trajectory-prediction
pip install -e ".[dev]"
```

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA-capable GPU (optional, for training)

---

## Usage

### Basic Training
```python
from trajectory_prediction import Config, TrajectoryTrainer

# Load configuration
config = Config.from_yaml('config/config.yaml')

# Initialize trainer
trainer = TrajectoryTrainer(config)

# Train model
trainer.train(train_loader, val_loader)
```

### Inference
```python
from trajectory_prediction import TrajectoryPredictor

# Load trained model
predictor = TrajectoryPredictor('model_path.pt', config)

# Make predictions
predictions = predictor.predict(input_features)
```

---

## Project Structure

```
src/
├── config/          # Configuration management
├── data/            # Data processing and feature extraction
├── models/          # Neural network models
├── training/        # Training pipeline
└── utils/           # Utility functions and logging
```

---

## Configuration

The project uses YAML configuration files. Example configuration:
```yaml
model:
  input_size: 60
  hidden_size: 128
  num_layers: 2
  dropout: 0.1

training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.001
```

See `config.yaml` for full configuration options.

---

## Development

### Setup Development Environment
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
```

---

## Docker Support

Build and run using Docker:
```bash
# Build training image
docker build -t trajectory-prediction:latest -f Dockerfile --target training .

# Run training
docker run --gpus all trajectory-prediction:latest train-trajectory
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Citation

If you use this code in your research, please cite:
```bibtex
@software{trajectory_prediction_lstm,
  title = {Trajectory Prediction LSTM},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/sexy-yume/trajectory-prediction}
}
```

---

## Acknowledgments

This project was developed using PyTorch  
Thanks to all contributors and users

---

## Contact

Your Name - [amwl1234@gmail.com](mailto:amwl1234@gmail.com)  
Project Link: [https://github.com/sexy-yume/trajectory-prediction](https://github.com/yourusername/trajectory-prediction)
