#!/usr/bin/env python3
"""
Professional Dataset Management for Helmet Detection
Training Engineer's Dataset Preparation Tool
"""

import os
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class ProfessionalDatasetManager:
    """
    Professional-grade dataset management for helmet detection training.
    """
    
    def __init__(self, dataset_root: str = "datasets"):
        self.dataset_root = Path(dataset_root)
        self.setup_directories()
        
        # Class definitions
        self.classes = {
            0: "helmet",
            1: "no_helmet"
        }
        
        # Quality metrics
        self.quality_metrics = {
            "min_images_per_class": 100,
            "max_class_imbalance": 2.0,  # Max ratio between classes
            "min_annotation_area": 0.001,  # Minimum bounding box area
            "max_annotation_area": 0.3     # Maximum bounding box area
        }
    
    def setup_directories(self):
        """Create professional dataset directory structure."""
        directories = [
            "raw_data/images",
            "raw_data/labels", 
            "processed_data/images/train",
            "processed_data/images/val",
            "processed_data/images/test",
            "processed_data/labels/train",
            "processed_data/labels/val", 
            "processed_data/labels/test",
            "quality_reports",
            "augmented_data"
        ]
        
        for directory in directories:
            (self.dataset_root / directory).mkdir(parents=True, exist_ok=True)
        
        print("✓ Professional dataset structure created")
    
    def validate_image_quality(self, image_path: str) -> Dict:
        """
        Validate image quality for training.
        
        Returns:
            Dict with quality metrics and recommendations
        """
        img = cv2.imread(image_path)
        if img is None:
            return {"valid": False, "error": "Cannot read image"}
        
        height, width = img.shape[:2]
        
        # Quality checks
        quality_report = {
            "valid": True,
            "resolution": f"{width}x{height}",
            "aspect_ratio": width / height,
            "file_size_mb": os.path.getsize(image_path) / (1024 * 1024),
            "recommendations": []
        }
        
        # Resolution check
        if width < 320 or height < 240:
            quality_report["recommendations"].append("Low resolution - consider higher quality images")
        
        # Aspect ratio check
        if quality_report["aspect_ratio"] < 0.5 or quality_report["aspect_ratio"] > 2.0:
            quality_report["recommendations"].append("Unusual aspect ratio - may affect training")
        
        # File size check
        if quality_report["file_size_mb"] > 10:
            quality_report["recommendations"].append("Large file size - consider compression")
        
        return quality_report
    
    def validate_annotations(self, label_path: str, image_path: str) -> Dict:
        """
        Validate annotation quality.
        
        Returns:
            Dict with annotation quality metrics
        """
        if not os.path.exists(label_path):
            return {"valid": False, "error": "Label file not found"}
        
        img = cv2.imread(image_path)
        if img is None:
            return {"valid": False, "error": "Cannot read corresponding image"}
        
        height, width = img.shape[:2]
        
        validation_report = {
            "valid": True,
            "num_annotations": 0,
            "class_distribution": {0: 0, 1: 0},
            "bbox_areas": [],
            "recommendations": []
        }
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        validation_report["recommendations"].append("Invalid annotation format")
                        continue
                    
                    class_id = int(parts[0])
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                    
                    # Validate class ID
                    if class_id not in [0, 1]:
                        validation_report["recommendations"].append(f"Invalid class ID: {class_id}")
                        continue
                    
                    validation_report["class_distribution"][class_id] += 1
                    validation_report["num_annotations"] += 1
                    
                    # Calculate bounding box area
                    area = bbox_width * bbox_height
                    validation_report["bbox_areas"].append(area)
                    
                    # Check bounding box size
                    if area < self.quality_metrics["min_annotation_area"]:
                        validation_report["recommendations"].append("Very small bounding box detected")
                    elif area > self.quality_metrics["max_annotation_area"]:
                        validation_report["recommendations"].append("Very large bounding box detected")
                    
                    # Check coordinates
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        validation_report["recommendations"].append("Invalid center coordinates")
                    
                    if not (0 <= bbox_width <= 1 and 0 <= bbox_height <= 1):
                        validation_report["recommendations"].append("Invalid bounding box dimensions")
        
        except Exception as e:
            validation_report["valid"] = False
            validation_report["error"] = str(e)
        
        return validation_report
    
    def generate_dataset_report(self) -> Dict:
        """
        Generate comprehensive dataset quality report.
        """
        report = {
            "dataset_overview": {},
            "class_distribution": {},
            "quality_issues": [],
            "recommendations": []
        }
        
        # Analyze each split
        for split in ["train", "val", "test"]:
            images_dir = self.dataset_root / "processed_data" / "images" / split
            labels_dir = self.dataset_root / "processed_data" / "labels" / split
            
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            label_files = list(labels_dir.glob("*.txt"))
            
            report["dataset_overview"][split] = {
                "images": len(image_files),
                "labels": len(label_files),
                "missing_labels": len(image_files) - len(label_files)
            }
            
            # Analyze class distribution
            class_counts = {0: 0, 1: 0}
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                class_id = int(line.split()[0])
                                if class_id in class_counts:
                                    class_counts[class_id] += 1
                except:
                    continue
            
            report["class_distribution"][split] = class_counts
        
        # Generate recommendations
        total_images = sum(data["images"] for data in report["dataset_overview"].values())
        if total_images < 200:
            report["recommendations"].append("Dataset too small - collect more images")
        
        # Check class balance
        total_helmet = sum(data.get(0, 0) for data in report["class_distribution"].values())
        total_no_helmet = sum(data.get(1, 0) for data in report["class_distribution"].values())
        
        if total_helmet > 0 and total_no_helmet > 0:
            imbalance_ratio = max(total_helmet, total_no_helmet) / min(total_helmet, total_no_helmet)
            if imbalance_ratio > self.quality_metrics["max_class_imbalance"]:
                report["recommendations"].append(f"Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        
        return report
    
    def create_yolo_config(self) -> str:
        """Create YOLO dataset configuration file."""
        config = {
            'path': str(self.dataset_root.absolute()),
            'train': 'processed_data/images/train',
            'val': 'processed_data/images/val',
            'test': 'processed_data/images/test',
            'nc': len(self.classes),
            'names': list(self.classes.values())
        }
        
        config_path = self.dataset_root / "data.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(config_path)
    
    def visualize_dataset(self):
        """Create dataset visualization plots."""
        report = self.generate_dataset_report()
        
        # Class distribution plot
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Dataset split distribution
        plt.subplot(2, 2, 1)
        splits = list(report["dataset_overview"].keys())
        images = [report["dataset_overview"][split]["images"] for split in splits]
        plt.bar(splits, images)
        plt.title("Images per Split")
        plt.ylabel("Number of Images")
        
        # Subplot 2: Class distribution
        plt.subplot(2, 2, 2)
        total_helmet = sum(data.get(0, 0) for data in report["class_distribution"].values())
        total_no_helmet = sum(data.get(1, 0) for data in report["class_distribution"].values())
        plt.pie([total_helmet, total_no_helmet], labels=["Helmet", "No Helmet"], autopct='%1.1f%%')
        plt.title("Overall Class Distribution")
        
        # Subplot 3: Missing labels
        plt.subplot(2, 2, 3)
        missing = [report["dataset_overview"][split]["missing_labels"] for split in splits]
        plt.bar(splits, missing, color='red', alpha=0.7)
        plt.title("Missing Labels per Split")
        plt.ylabel("Missing Labels")
        
        # Subplot 4: Recommendations
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.9, "Dataset Quality Report", fontsize=14, fontweight='bold')
        y_pos = 0.8
        for rec in report["recommendations"]:
            plt.text(0.1, y_pos, f"• {rec}", fontsize=10)
            y_pos -= 0.1
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.dataset_root / "quality_reports" / "dataset_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Dataset visualization saved to quality_reports/dataset_analysis.png")


def main():
    """Main function for professional dataset management."""
    print("Professional Helmet Detection Dataset Manager")
    print("=" * 60)
    
    dm = ProfessionalDatasetManager()
    
    while True:
        print("\nProfessional Dataset Management Options:")
        print("1. Validate image quality")
        print("2. Validate annotations")
        print("3. Generate dataset report")
        print("4. Create dataset visualization")
        print("5. Create YOLO config")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if image_path:
                report = dm.validate_image_quality(image_path)
                print(f"Quality Report: {json.dumps(report, indent=2)}")
        
        elif choice == "2":
            image_path = input("Enter image path: ").strip()
            label_path = input("Enter label path: ").strip()
            if image_path and label_path:
                report = dm.validate_annotations(label_path, image_path)
                print(f"Annotation Report: {json.dumps(report, indent=2)}")
        
        elif choice == "3":
            report = dm.generate_dataset_report()
            print(f"Dataset Report: {json.dumps(report, indent=2)}")
        
        elif choice == "4":
            dm.visualize_dataset()
        
        elif choice == "5":
            config_path = dm.create_yolo_config()
            print(f"YOLO config created: {config_path}")
        
        elif choice == "6":
            break
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
