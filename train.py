"""
Training script for Vision Transformer (ViT) model on PlantNet300K dataset.

This script fine-tunes google/vit-base-patch16-224 model from Hugging Face
on the mikehemberger/plantnet300K dataset for plant image classification.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    Resize,
    CenterCrop
)
import numpy as np
from evaluate import load as load_metric


def load_plantnet_dataset():
    """Load the PlantNet300K dataset from Hugging Face."""
    print("Loading PlantNet300K dataset...")
    dataset = load_dataset("mikehemberger/plantnet300K")
    return dataset


def get_label_mappings(dataset):
    """Create label mappings from the dataset."""
    # Get unique labels from training set
    if 'train' in dataset:
        labels = dataset['train'].features['label']
        if hasattr(labels, 'names'):
            label_names = labels.names
        else:
            # If labels is ClassLabel type
            label_names = labels.names if hasattr(labels, 'names') else sorted(set(dataset['train']['label']))
    else:
        # Fallback if no train split
        split_name = list(dataset.keys())[0]
        label_names = sorted(set(dataset[split_name]['label']))
    
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return label2id, id2label


def create_transforms(image_processor, is_train=True):
    """Create image transformation pipeline."""
    if is_train:
        return Compose([
            RandomResizedCrop(image_processor.size['height']),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ])
    else:
        return Compose([
            Resize(image_processor.size['height']),
            CenterCrop(image_processor.size['height']),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ])


def preprocess_data(examples, image_processor, transforms):
    """Preprocess images for the model."""
    examples['pixel_values'] = [
        transforms(image.convert("RGB")) for image in examples['image']
    ]
    return examples


def compute_metrics(eval_pred):
    """Compute accuracy and other metrics for evaluation."""
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    """Main training function."""
    # Configuration
    MODEL_NAME = "google/vit-base-patch16-224"
    OUTPUT_DIR = "./vit-plantnet-model"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = load_plantnet_dataset()
    print(f"Dataset splits: {dataset.keys()}")
    
    # Get label mappings
    label2id, id2label = get_label_mappings(dataset)
    num_labels = len(label2id)
    print(f"Number of classes: {num_labels}")
    
    # Load image processor
    print(f"Loading image processor from {MODEL_NAME}...")
    image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    
    # Create transforms
    train_transforms = create_transforms(image_processor, is_train=True)
    val_transforms = create_transforms(image_processor, is_train=False)
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    if 'train' in dataset:
        train_dataset = dataset['train'].with_transform(
            lambda x: preprocess_data(x, image_processor, train_transforms)
        )
    else:
        # If no explicit train split, use the first available split
        split_name = list(dataset.keys())[0]
        train_dataset = dataset[split_name].with_transform(
            lambda x: preprocess_data(x, image_processor, train_transforms)
        )
    
    # Handle validation/test split
    if 'validation' in dataset:
        eval_dataset = dataset['validation'].with_transform(
            lambda x: preprocess_data(x, image_processor, val_transforms)
        )
    elif 'test' in dataset:
        eval_dataset = dataset['test'].with_transform(
            lambda x: preprocess_data(x, image_processor, val_transforms)
        )
    else:
        # Split train set if no validation set exists
        print("No validation set found, splitting training data...")
        split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
    
    # Load model
    print(f"Loading model {MODEL_NAME}...")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
    )
    
    # Data collator
    data_collator = DefaultDataCollator()
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=image_processor,  # Save image processor with model
    )
    
    # Train the model
    print("Starting training...")
    train_result = trainer.train()
    
    # Save the final model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    trainer.save_state()
    
    # Evaluate the model
    print("Evaluating model...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    # Print training results
    print("\nTraining completed!")
    print(f"Training metrics: {train_result.metrics}")
    print(f"Evaluation metrics: {metrics}")
    
    return trainer, metrics


if __name__ == "__main__":
    main()
