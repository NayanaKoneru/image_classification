# Image Classification with Vision Transformer

Image classification using Vision Transformer (ViT) trained on PlantNet-300K dataset to classify different plant species. This project uses the `google/vit-base-patch16-224` model and `mikehemberger/plantnet300K` dataset from Hugging Face.

## Overview

This project fine-tunes a pre-trained Vision Transformer (ViT) model for plant species classification using the PlantNet300K dataset containing diverse plant images.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster training)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/NayanaKoneru/image_classification.git
cd image_classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script:
```bash
python train.py
```

The script will:
- Download the `google/vit-base-patch16-224` model from Hugging Face
- Load the `mikehemberger/plantnet300K` dataset
- Fine-tune the model on the plant classification task
- Save the trained model to `./vit-plantnet-model`

### Training Configuration

You can modify training parameters in `train.py`:
- `BATCH_SIZE`: Batch size per device (default: 32)
- `NUM_EPOCHS`: Number of training epochs (default: 10)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `OUTPUT_DIR`: Output directory for saved models (default: ./vit-plantnet-model)

## Model Details

- **Base Model**: `google/vit-base-patch16-224`
- **Dataset**: `mikehemberger/plantnet300K`
- **Task**: Multi-class plant species classification
- **Image Size**: 224x224 pixels
- **Training Features**:
  - Data augmentation (random crop, horizontal flip)
  - Mixed precision training (FP16)
  - Learning rate warmup
  - Best model checkpoint saving

## Output

After training, you'll find:
- Trained model weights in `./vit-plantnet-model`
- Training logs in `./vit-plantnet-model/logs`
- Evaluation metrics

## License

See LICENSE file for details.
