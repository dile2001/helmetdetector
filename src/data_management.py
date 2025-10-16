"""
Helmet Detection System - Data Management and Training Utilities

This module provides utilities for dataset management, model training, and evaluation.
Supports YOLO format datasets and provides training scripts.
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import zipfile

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages helmet detection datasets in YOLO format.
    """
    
    def __init__(self, dataset_path: str = "datasets"):
        """
        Initialize dataset manager.
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset structure
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"
        
        # Create subdirectories
        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Class names
        self.class_names = ["helmet", "no_helmet"]
        self.class_ids = {name: i for i, name in enumerate(self.class_names)}
    
    def create_yaml_config(self) -> str:
        """Create YOLO dataset configuration file."""
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        config_path = self.dataset_path / "data.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created dataset config: {config_path}")
        return str(config_path)
    
    def add_image_with_labels(self, image_path: str, labels: List[Dict], split: str = "train"):
        """
        Add image and labels to dataset.
        
        Args:
            image_path: Path to image file
            labels: List of label dictionaries with 'class', 'bbox' (x1,y1,x2,y2)
            split: Dataset split (train/val/test)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return
        
        # Copy image
        image_filename = image_path.name
        dest_image_path = self.images_dir / split / image_filename
        shutil.copy2(image_path, dest_image_path)
        
        # Create label file
        label_filename = image_path.stem + ".txt"
        label_path = self.labels_dir / split / label_filename
        
        with open(label_path, 'w') as f:
            for label in labels:
                class_name = label['class']
                if class_name not in self.class_ids:
                    logger.warning(f"Unknown class: {class_name}")
                    continue
                
                class_id = self.class_ids[class_name]
                bbox = label['bbox']  # x1, y1, x2, y2
                
                # Convert to YOLO format (normalized center coordinates)
                img = cv2.imread(str(dest_image_path))
                img_height, img_width = img.shape[:2]
                
                x_center = (bbox[0] + bbox[2]) / 2 / img_width
                y_center = (bbox[1] + bbox[3]) / 2 / img_height
                width = (bbox[2] - bbox[0]) / img_width
                height = (bbox[3] - bbox[1]) / img_height
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        logger.info(f"Added {image_filename} to {split} split")
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """
        Split dataset into train/val/test splits.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
        """
        # Get all images
        all_images = list(self.images_dir.glob("**/*.jpg")) + list(self.images_dir.glob("**/*.png"))
        
        if not all_images:
            logger.warning("No images found in dataset")
            return
        
        # Split images
        train_images, temp_images = train_test_split(
            all_images, test_size=(val_ratio + test_ratio), random_state=42
        )
        val_images, test_images = train_test_split(
            temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
        )
        
        # Move images and labels to appropriate splits
        for split_name, images in [("train", train_images), ("val", val_images), ("test", test_images)]:
            for img_path in images:
                # Move image
                dest_img_path = self.images_dir / split_name / img_path.name
                shutil.move(str(img_path), str(dest_img_path))
                
                # Move corresponding label
                label_path = self.labels_dir / img_path.stem / (img_path.stem + ".txt")
                if label_path.exists():
                    dest_label_path = self.labels_dir / split_name / (img_path.stem + ".txt")
                    shutil.move(str(label_path), str(dest_label_path))
        
        logger.info(f"Dataset split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics."""
        stats = {}
        
        for split in ["train", "val", "test"]:
            images_dir = self.images_dir / split
            labels_dir = self.labels_dir / split
            
            image_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
            label_count = len(list(labels_dir.glob("*.txt")))
            
            # Count labels per class
            class_counts = {name: 0 for name in self.class_names}
            for label_file in labels_dir.glob("*.txt"):
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        if class_id < len(self.class_names):
                            class_counts[self.class_names[class_id]] += 1
            
            stats[split] = {
                'images': image_count,
                'labels': label_count,
                'class_counts': class_counts
            }
        
        return stats
    
    def download_sample_dataset(self):
        """Download a sample helmet detection dataset."""
        # This would download a sample dataset from a public source
        # For now, we'll create a placeholder
        logger.info("Sample dataset download not implemented yet")
        logger.info("Please add your own helmet detection images and labels")


class ModelTrainer:
    """
    Handles model training and evaluation.
    """
    
    def __init__(self, dataset_manager: DatasetManager):
        """
        Initialize model trainer.
        
        Args:
            dataset_manager: Dataset manager instance
        """
        self.dataset_manager = dataset_manager
        self.model = None
        self.training_results = None
    
    def prepare_training(self, model_size: str = "n") -> str:
        """
        Prepare training configuration.
        
        Args:
            model_size: YOLO model size (n, s, m, l, x)
            
        Returns:
            Path to dataset config file
        """
        # Create dataset config
        config_path = self.dataset_manager.create_yaml_config()
        
        # Load base model
        model_name = f"yolov8{model_size}.pt"
        self.model = YOLO(model_name)
        
        logger.info(f"Prepared training with {model_name}")
        return config_path
    
    def train(self, epochs: int = 50, imgsz: int = 640, batch_size: int = 16, 
              config_path: str = None) -> Dict:
        """
        Train the helmet detection model.
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size
            config_path: Path to dataset config
            
        Returns:
            Training results dictionary
        """
        if not self.model:
            raise RuntimeError("Model not prepared. Call prepare_training() first.")
        
        if not config_path:
            config_path = self.dataset_manager.create_yaml_config()
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        # Train the model
        self.training_results = self.model.train(
            data=config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device='cpu',  # Use 'cuda' if GPU available
            project='runs/detect',
            name='helmet_detection',
            save=True,
            plots=True
        )
        
        logger.info("Training completed!")
        
        # Get best model path
        best_model_path = self.training_results.save_dir / "weights" / "best.pt"
        
        return {
            'success': True,
            'best_model_path': str(best_model_path),
            'results_dir': str(self.training_results.save_dir),
            'metrics': self._extract_metrics()
        }
    
    def _extract_metrics(self) -> Dict:
        """Extract training metrics from results."""
        if not self.training_results:
            return {}
        
        # This would extract metrics from the training results
        # For now, return placeholder
        return {
            'mAP50': 0.0,
            'mAP50-95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    def evaluate(self, model_path: str, test_data: str = None) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            model_path: Path to trained model
            test_data: Path to test dataset config
            
        Returns:
            Evaluation results
        """
        if not test_data:
            test_data = self.dataset_manager.create_yaml_config()
        
        model = YOLO(model_path)
        results = model.val(data=test_data)
        
        return {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr
        }
    
    def export_model(self, model_path: str, format: str = "onnx") -> str:
        """
        Export model to different formats.
        
        Args:
            model_path: Path to trained model
            format: Export format (onnx, tflite, etc.)
            
        Returns:
            Path to exported model
        """
        model = YOLO(model_path)
        exported_path = model.export(format=format)
        
        logger.info(f"Model exported to {exported_path}")
        return exported_path


class DataAugmentation:
    """
    Data augmentation utilities for improving model performance.
    """
    
    @staticmethod
    def augment_image(image_path: str, output_dir: str, num_augmentations: int = 5):
        """
        Apply data augmentation to an image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save augmented images
            num_augmentations: Number of augmented versions to create
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(image_path).stem
        
        for i in range(num_augmentations):
            augmented = image.copy()
            
            # Apply random augmentations
            if np.random.random() > 0.5:
                augmented = cv2.flip(augmented, 1)  # Horizontal flip
            
            if np.random.random() > 0.5:
                # Random brightness adjustment
                alpha = np.random.uniform(0.8, 1.2)
                augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=0)
            
            if np.random.random() > 0.5:
                # Random rotation
                angle = np.random.uniform(-15, 15)
                h, w = augmented.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                augmented = cv2.warpAffine(augmented, matrix, (w, h))
            
            # Save augmented image
            output_path = output_dir / f"{base_name}_aug_{i}.jpg"
            cv2.imwrite(str(output_path), augmented)
        
        logger.info(f"Created {num_augmentations} augmented versions of {image_path}")


class DatasetValidator:
    """
    Validates dataset integrity and format.
    """
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
    
    def validate_dataset(self) -> Dict:
        """
        Validate the entire dataset.
        
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check dataset structure
        if not self.dataset_manager.images_dir.exists():
            results['errors'].append("Images directory not found")
            results['valid'] = False
        
        if not self.dataset_manager.labels_dir.exists():
            results['errors'].append("Labels directory not found")
            results['valid'] = False
        
        # Validate each split
        for split in ["train", "val", "test"]:
            split_results = self._validate_split(split)
            results['stats'][split] = split_results
            
            if not split_results['valid']:
                results['valid'] = False
                results['errors'].extend(split_results['errors'])
            
            results['warnings'].extend(split_results['warnings'])
        
        return results
    
    def _validate_split(self, split: str) -> Dict:
        """Validate a specific dataset split."""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'image_count': 0,
            'label_count': 0,
            'missing_labels': [],
            'invalid_labels': []
        }
        
        images_dir = self.dataset_manager.images_dir / split
        labels_dir = self.dataset_manager.labels_dir / split
        
        # Count images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        results['image_count'] = len(image_files)
        
        # Count labels
        label_files = list(labels_dir.glob("*.txt"))
        results['label_count'] = len(label_files)
        
        # Check for missing labels
        for img_file in image_files:
            label_file = labels_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                results['missing_labels'].append(img_file.name)
                results['warnings'].append(f"Missing label for {img_file.name}")
        
        # Validate label files
        for label_file in label_files:
            if not self._validate_label_file(label_file):
                results['invalid_labels'].append(label_file.name)
                results['errors'].append(f"Invalid label file: {label_file.name}")
                results['valid'] = False
        
        return results
    
    def _validate_label_file(self, label_path: Path) -> bool:
        """Validate a single label file."""
        try:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        logger.error(f"Invalid label format in {label_path}:{line_num}")
                        return False
                    
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        
                        if class_id < 0 or class_id >= len(self.dataset_manager.class_names):
                            logger.error(f"Invalid class ID {class_id} in {label_path}:{line_num}")
                            return False
                        
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            logger.error(f"Invalid coordinates in {label_path}:{line_num}")
                            return False
                    
                    except ValueError:
                        logger.error(f"Invalid number format in {label_path}:{line_num}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error reading label file {label_path}: {e}")
            return False


if __name__ == "__main__":
    # Test dataset manager
    dataset_manager = DatasetManager()
    print("Dataset manager initialized")
    
    # Create sample config
    config_path = dataset_manager.create_yaml_config()
    print(f"Config created: {config_path}")
    
    # Get stats
    stats = dataset_manager.get_dataset_stats()
    print(f"Dataset stats: {stats}")
