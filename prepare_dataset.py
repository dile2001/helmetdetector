#!/usr/bin/env python3
"""
Dataset Preparation Helper

This script helps you prepare your helmet detection dataset.
"""

import os
import shutil
from pathlib import Path
import cv2
import json


def add_image_to_dataset(image_path, labels, split="train"):
    """
    Add an image and its labels to the dataset.
    
    Args:
        image_path: Path to the image file
        labels: List of label dictionaries
                [{"class": "helmet", "bbox": [x1, y1, x2, y2]}, ...]
        split: Dataset split (train/val/test)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    # Copy image to dataset
    dest_image = Path(f"datasets/images/{split}") / image_path.name
    shutil.copy2(image_path, dest_image)
    
    # Create label file
    label_file = Path(f"datasets/labels/{split}") / (image_path.stem + ".txt")
    
    # Load image to get dimensions
    img = cv2.imread(str(dest_image))
    img_height, img_width = img.shape[:2]
    
    with open(label_file, 'w') as f:
        for label in labels:
            class_name = label['class']
            bbox = label['bbox']  # [x1, y1, x2, y2]
            
            # Convert to YOLO format
            if class_name == "helmet":
                class_id = 0
            elif class_name == "no_helmet":
                class_id = 1
            else:
                print(f"Warning: Unknown class {class_name}")
                continue
            
            # Normalize coordinates
            x_center = (bbox[0] + bbox[2]) / 2 / img_width
            y_center = (bbox[1] + bbox[3]) / 2 / img_height
            width = (bbox[2] - bbox[0]) / img_width
            height = (bbox[3] - bbox[1]) / img_height
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"✓ Added {image_path.name} to {split} split")
    return True


def split_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train/val/test splits."""
    from src.data_management import DatasetManager
    
    dm = DatasetManager("datasets")
    dm.split_dataset(train_ratio, val_ratio, test_ratio)
    print("✓ Dataset split completed")


def get_dataset_stats():
    """Get current dataset statistics."""
    from src.data_management import DatasetManager
    
    dm = DatasetManager("datasets")
    stats = dm.get_dataset_stats()
    
    print("\nDataset Statistics:")
    print("=" * 50)
    
    total_images = 0
    total_labels = 0
    
    for split, data in stats.items():
        images = data['images']
        labels = data['labels']
        total_images += images
        total_labels += labels
        
        print(f"{split.upper()}:")
        print(f"  Images: {images}")
        print(f"  Labels: {labels}")
        
        if 'class_counts' in data:
            print(f"  Classes:")
            for class_name, count in data['class_counts'].items():
                print(f"    {class_name}: {count}")
        print()
    
    print(f"TOTAL: {total_images} images, {total_labels} labels")
    
    # Recommendations
    if total_images < 50:
        print("\n⚠️  WARNING: Very small dataset!")
        print("   Consider adding more images for better results.")
    elif total_images < 100:
        print("\n⚠️  WARNING: Small dataset!")
        print("   Results may be limited. Add more images if possible.")
    elif total_images >= 200:
        print("\n✅ Good dataset size!")
        print("   Should produce decent results.")
    
    return stats


def main():
    """Main function for dataset preparation."""
    print("Helmet Detection Dataset Preparation")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Add image to dataset")
        print("2. Split dataset")
        print("3. Show dataset statistics")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if not image_path:
                continue
            
            print("\nEnter labels (press Enter when done):")
            labels = []
            while True:
                class_name = input("Class (helmet/no_helmet): ").strip()
                if not class_name:
                    break
                
                try:
                    x1 = float(input("x1: "))
                    y1 = float(input("y1: "))
                    x2 = float(input("x2: "))
                    y2 = float(input("y2: "))
                    
                    labels.append({
                        "class": class_name,
                        "bbox": [x1, y1, x2, y2]
                    })
                except ValueError:
                    print("Invalid coordinates!")
                    continue
            
            split = input("Split (train/val/test) [train]: ").strip() or "train"
            
            if labels:
                add_image_to_dataset(image_path, labels, split)
            else:
                print("No labels provided!")
        
        elif choice == "2":
            split_dataset()
        
        elif choice == "3":
            get_dataset_stats()
        
        elif choice == "4":
            break
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
