#!/usr/bin/env python3
"""
Helmet Detection System - Training Script

This script handles model training with various options.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_management import DatasetManager, ModelTrainer, DatasetValidator


def setup_logging():
    """Setup logging for training."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_dataset(dataset_path: str):
    """Validate the dataset before training."""
    print("Validating dataset...")
    
    dataset_manager = DatasetManager(dataset_path)
    validator = DatasetValidator(dataset_manager)
    
    results = validator.validate_dataset()
    
    if results['valid']:
        print("✓ Dataset validation passed")
        
        # Print stats
        for split, stats in results['stats'].items():
            print(f"  {split}: {stats['image_count']} images, {stats['label_count']} labels")
            for class_name, count in stats['class_counts'].items():
                print(f"    {class_name}: {count}")
    else:
        print("✗ Dataset validation failed:")
        for error in results['errors']:
            print(f"  Error: {error}")
        for warning in results['warnings']:
            print(f"  Warning: {warning}")
        
        return False
    
    return True


def train_model(dataset_path: str, model_size: str, epochs: int, batch_size: int, 
                imgsz: int, device: str):
    """Train the helmet detection model."""
    print(f"Starting training with YOLOv8{model_size}...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}")
    
    # Initialize components
    dataset_manager = DatasetManager(dataset_path)
    trainer = ModelTrainer(dataset_manager)
    
    # Prepare training
    config_path = trainer.prepare_training(model_size)
    
    # Train model
    results = trainer.train(
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        config_path=config_path
    )
    
    if results['success']:
        print("✓ Training completed successfully!")
        print(f"Best model saved to: {results['best_model_path']}")
        print(f"Results directory: {results['results_dir']}")
        
        # Print metrics
        metrics = results['metrics']
        print("\nTraining Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return results['best_model_path']
    else:
        print("✗ Training failed!")
        return None


def evaluate_model(model_path: str, dataset_path: str):
    """Evaluate the trained model."""
    print(f"Evaluating model: {model_path}")
    
    dataset_manager = DatasetManager(dataset_path)
    trainer = ModelTrainer(dataset_manager)
    
    results = trainer.evaluate(model_path)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    return results


def export_model(model_path: str, format: str):
    """Export model to different formats."""
    print(f"Exporting model to {format} format...")
    
    dataset_manager = DatasetManager()
    trainer = ModelTrainer(dataset_manager)
    
    exported_path = trainer.export_model(model_path, format)
    print(f"✓ Model exported to: {exported_path}")
    
    return exported_path


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Helmet Detection Model")
    parser.add_argument("--dataset", type=str, default="datasets",
                       help="Path to dataset directory")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], 
                       default="n", help="YOLO model size")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size for training")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for training (cpu/cuda)")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate dataset, don't train")
    parser.add_argument("--evaluate", type=str,
                       help="Path to model to evaluate")
    parser.add_argument("--export", type=str,
                       help="Path to model to export")
    parser.add_argument("--export-format", choices=["onnx", "tflite", "engine"],
                       default="onnx", help="Export format")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    try:
        if args.validate_only:
            # Only validate dataset
            if validate_dataset(args.dataset):
                print("Dataset is ready for training!")
            else:
                print("Please fix dataset issues before training.")
                sys.exit(1)
        
        elif args.evaluate:
            # Evaluate existing model
            if not Path(args.evaluate).exists():
                print(f"Model not found: {args.evaluate}")
                sys.exit(1)
            
            evaluate_model(args.evaluate, args.dataset)
        
        elif args.export:
            # Export existing model
            if not Path(args.export).exists():
                print(f"Model not found: {args.export}")
                sys.exit(1)
            
            export_model(args.export, args.export_format)
        
        else:
            # Full training pipeline
            print("=" * 60)
            print("Helmet Detection Model Training")
            print("=" * 60)
            
            # Validate dataset
            if not validate_dataset(args.dataset):
                print("Dataset validation failed. Please fix issues before training.")
                sys.exit(1)
            
            # Train model
            model_path = train_model(
                args.dataset, args.model_size, args.epochs, 
                args.batch_size, args.imgsz, args.device
            )
            
            if model_path:
                # Evaluate model
                print("\n" + "=" * 40)
                print("Model Evaluation")
                print("=" * 40)
                evaluate_model(model_path, args.dataset)
                
                # Export model
                print("\n" + "=" * 40)
                print("Model Export")
                print("=" * 40)
                export_model(model_path, "onnx")
                
                print("\n✓ Training pipeline completed successfully!")
                print(f"Your trained model is ready: {model_path}")
            else:
                print("Training failed!")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
