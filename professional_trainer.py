#!/usr/bin/env python3
"""
Professional Helmet Detection Model Training
Training Engineer's Training Configuration and Execution
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import seaborn as sns


class ProfessionalModelTrainer:
    """
    Professional-grade model training for helmet detection.
    """
    
    def __init__(self, dataset_path: str = "datasets"):
        self.dataset_path = Path(dataset_path)
        self.training_config = self.load_training_config()
        self.setup_logging()
        
        # Training metrics tracking
        self.training_history = {
            "epochs": [],
            "losses": {"box_loss": [], "cls_loss": [], "dfl_loss": []},
            "metrics": {"mAP50": [], "mAP50-95": [], "precision": [], "recall": []}
        }
    
    def setup_logging(self):
        """Setup professional logging."""
        log_dir = Path("logs/training")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{int(time.time())}.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_training_config(self) -> Dict:
        """Load professional training configuration."""
        config_file = Path("config/training_config.yaml")
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default professional configuration
            default_config = {
                "model": {
                    "size": "n",  # nano for Pi 4, s/m/l for desktop
                    "pretrained": True,
                    "freeze_layers": None
                },
                "training": {
                    "epochs": 100,
                    "batch_size": 16,
                    "image_size": 640,
                    "learning_rate": 0.01,
                    "momentum": 0.937,
                    "weight_decay": 0.0005,
                    "patience": 50,
                    "save_period": 10
                },
                "data_augmentation": {
                    "mosaic": 1.0,
                    "mixup": 0.0,
                    "copy_paste": 0.0,
                    "degrees": 0.0,
                    "translate": 0.1,
                    "scale": 0.5,
                    "shear": 0.0,
                    "perspective": 0.0,
                    "flipud": 0.0,
                    "fliplr": 0.5,
                    "hsv_h": 0.015,
                    "hsv_s": 0.7,
                    "hsv_v": 0.4
                },
                "validation": {
                    "val": True,
                    "plots": True,
                    "save_json": True,
                    "save_hybrid": False
                },
                "optimization": {
                    "device": "auto",  # auto, cpu, cuda, 0, 1, etc.
                    "workers": 8,
                    "project": "runs/detect",
                    "name": "helmet_detection_professional",
                    "exist_ok": False,
                    "pretrained": True,
                    "optimizer": "auto",  # auto, SGD, Adam, AdamW, RMSProp
                    "close_mosaic": 10
                }
            }
            
            # Save default config
            config_file.parent.mkdir(exist_ok=True)
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            return default_config
    
    def validate_training_setup(self) -> Dict:
        """
        Validate training setup and environment.
        
        Returns:
            Dict with validation results and recommendations
        """
        validation_report = {
            "valid": True,
            "issues": [],
            "recommendations": [],
            "system_info": {}
        }
        
        # Check dataset
        data_yaml = self.dataset_path / "data.yaml"
        if not data_yaml.exists():
            validation_report["issues"].append("Dataset config file not found")
            validation_report["valid"] = False
        
        # Check system resources
        if torch.cuda.is_available():
            validation_report["system_info"]["gpu"] = {
                "available": True,
                "count": torch.cuda.device_count(),
                "current": torch.cuda.current_device(),
                "name": torch.cuda.get_device_name()
            }
        else:
            validation_report["system_info"]["gpu"] = {"available": False}
            validation_report["recommendations"].append("Consider using GPU for faster training")
        
        # Check memory
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        validation_report["system_info"]["memory_gb"] = memory_gb
        
        if memory_gb < 8:
            validation_report["recommendations"].append("Low memory - consider reducing batch size")
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        validation_report["system_info"]["free_disk_gb"] = free_gb
        
        if free_gb < 10:
            validation_report["recommendations"].append("Low disk space - ensure sufficient space for training artifacts")
        
        return validation_report
    
    def optimize_training_config(self, target_device: str = "auto") -> Dict:
        """
        Optimize training configuration based on target device.
        
        Args:
            target_device: Target deployment device (pi4, desktop, gpu)
        """
        optimized_config = self.training_config.copy()
        
        if target_device == "pi4":
            # Raspberry Pi 4 optimizations
            optimized_config["model"]["size"] = "n"
            optimized_config["training"]["batch_size"] = 4
            optimized_config["training"]["image_size"] = 320
            optimized_config["training"]["epochs"] = 50
            optimized_config["optimization"]["workers"] = 2
            optimized_config["optimization"]["device"] = "cpu"
            
        elif target_device == "desktop":
            # Desktop optimizations
            optimized_config["model"]["size"] = "s"
            optimized_config["training"]["batch_size"] = 16
            optimized_config["training"]["image_size"] = 640
            optimized_config["training"]["epochs"] = 100
            optimized_config["optimization"]["workers"] = 8
            
        elif target_device == "gpu":
            # GPU optimizations
            optimized_config["model"]["size"] = "m"
            optimized_config["training"]["batch_size"] = 32
            optimized_config["training"]["image_size"] = 640
            optimized_config["training"]["epochs"] = 200
            optimized_config["optimization"]["workers"] = 16
        
        return optimized_config
    
    def train_model(self, config: Optional[Dict] = None, target_device: str = "auto") -> Dict:
        """
        Execute professional model training.
        
        Args:
            config: Custom training configuration
            target_device: Target deployment device
            
        Returns:
            Training results and metrics
        """
        # Use optimized config if not provided
        if config is None:
            config = self.optimize_training_config(target_device)
        
        # Validate setup
        validation = self.validate_training_setup()
        if not validation["valid"]:
            self.logger.error(f"Training setup validation failed: {validation['issues']}")
            return {"success": False, "error": "Setup validation failed"}
        
        self.logger.info("Starting professional model training")
        self.logger.info(f"Target device: {target_device}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        try:
            # Initialize model
            model_size = config["model"]["size"]
            model_name = f"yolov8{model_size}.pt"
            model = YOLO(model_name)
            
            # Prepare dataset config
            data_yaml = self.dataset_path / "data.yaml"
            if not data_yaml.exists():
                self.logger.error("Dataset config file not found")
                return {"success": False, "error": "Dataset config not found"}
            
            # Training parameters
            training_params = {
                "data": str(data_yaml),
                "epochs": config["training"]["epochs"],
                "batch": config["training"]["batch_size"],
                "imgsz": config["training"]["image_size"],
                "device": config["optimization"]["device"],
                "workers": config["optimization"]["workers"],
                "project": config["optimization"]["project"],
                "name": config["optimization"]["name"],
                "exist_ok": config["optimization"]["exist_ok"],
                "pretrained": config["optimization"]["pretrained"],
                "optimizer": config["optimization"]["optimizer"],
                "patience": config["training"]["patience"],
                "save_period": config["training"]["save_period"],
                "plots": config["validation"]["plots"],
                "val": config["validation"]["val"],
                "save_json": config["validation"]["save_json"],
                "close_mosaic": config["optimization"]["close_mosaic"]
            }
            
            # Data augmentation parameters
            aug_params = config["data_augmentation"]
            training_params.update(aug_params)
            
            # Start training
            start_time = time.time()
            results = model.train(**training_params)
            training_time = time.time() - start_time
            
            # Extract results
            best_model_path = results.save_dir / "weights" / "best.pt"
            last_model_path = results.save_dir / "weights" / "last.pt"
            
            # Get metrics
            metrics = self.extract_training_metrics(results)
            
            training_results = {
                "success": True,
                "training_time_hours": training_time / 3600,
                "best_model_path": str(best_model_path),
                "last_model_path": str(last_model_path),
                "results_dir": str(results.save_dir),
                "metrics": metrics,
                "config_used": config,
                "validation_report": validation
            }
            
            # Save training results
            self.save_training_results(training_results)
            
            self.logger.info("Training completed successfully")
            self.logger.info(f"Best model: {best_model_path}")
            self.logger.info(f"Training time: {training_time/3600:.2f} hours")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def extract_training_metrics(self, results) -> Dict:
        """Extract training metrics from results."""
        try:
            # This would extract metrics from the training results
            # For now, return placeholder metrics
            return {
                "mAP50": 0.0,
                "mAP50-95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "box_loss": 0.0,
                "cls_loss": 0.0,
                "dfl_loss": 0.0
            }
        except Exception as e:
            self.logger.warning(f"Could not extract metrics: {e}")
            return {}
    
    def save_training_results(self, results: Dict):
        """Save training results and configuration."""
        results_dir = Path("training_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"training_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Training results saved to {results_file}")
    
    def evaluate_model(self, model_path: str) -> Dict:
        """
        Evaluate trained model performance.
        
        Args:
            model_path: Path to trained model
            
        Returns:
            Evaluation results
        """
        try:
            model = YOLO(model_path)
            data_yaml = self.dataset_path / "data.yaml"
            
            results = model.val(data=str(data_yaml))
            
            evaluation_results = {
                "mAP50": results.box.map50,
                "mAP50-95": results.box.map,
                "precision": results.box.mp,
                "recall": results.box.mr,
                "f1_score": 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr)
            }
            
            self.logger.info(f"Model evaluation completed: {evaluation_results}")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return {"error": str(e)}
    
    def export_model(self, model_path: str, formats: List[str] = ["onnx", "tflite"]) -> Dict:
        """
        Export model to different formats for deployment.
        
        Args:
            model_path: Path to trained model
            formats: List of export formats
            
        Returns:
            Export results
        """
        export_results = {}
        
        try:
            model = YOLO(model_path)
            
            for format_type in formats:
                try:
                    exported_path = model.export(format=format_type)
                    export_results[format_type] = {
                        "success": True,
                        "path": str(exported_path)
                    }
                    self.logger.info(f"Model exported to {format_type}: {exported_path}")
                except Exception as e:
                    export_results[format_type] = {
                        "success": False,
                        "error": str(e)
                    }
                    self.logger.error(f"Export to {format_type} failed: {e}")
            
            return export_results
            
        except Exception as e:
            self.logger.error(f"Model export failed: {e}")
            return {"error": str(e)}


def main():
    """Main function for professional training."""
    print("Professional Helmet Detection Model Training")
    print("=" * 60)
    
    trainer = ProfessionalModelTrainer()
    
    while True:
        print("\nProfessional Training Options:")
        print("1. Validate training setup")
        print("2. Train model (Pi 4 optimized)")
        print("3. Train model (Desktop optimized)")
        print("4. Train model (GPU optimized)")
        print("5. Evaluate model")
        print("6. Export model")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            validation = trainer.validate_training_setup()
            print(f"Validation Report: {json.dumps(validation, indent=2)}")
        
        elif choice == "2":
            print("Training for Raspberry Pi 4 deployment...")
            results = trainer.train_model(target_device="pi4")
            print(f"Training Results: {json.dumps(results, indent=2)}")
        
        elif choice == "3":
            print("Training for Desktop deployment...")
            results = trainer.train_model(target_device="desktop")
            print(f"Training Results: {json.dumps(results, indent=2)}")
        
        elif choice == "4":
            print("Training for GPU deployment...")
            results = trainer.train_model(target_device="gpu")
            print(f"Training Results: {json.dumps(results, indent=2)}")
        
        elif choice == "5":
            model_path = input("Enter model path: ").strip()
            if model_path:
                evaluation = trainer.evaluate_model(model_path)
                print(f"Evaluation Results: {json.dumps(evaluation, indent=2)}")
        
        elif choice == "6":
            model_path = input("Enter model path: ").strip()
            if model_path:
                export_results = trainer.export_model(model_path)
                print(f"Export Results: {json.dumps(export_results, indent=2)}")
        
        elif choice == "7":
            break
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
