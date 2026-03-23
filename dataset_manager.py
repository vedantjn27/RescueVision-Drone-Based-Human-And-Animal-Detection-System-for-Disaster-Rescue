import os
import requests
import zipfile
from pathlib import Path
import yaml

class DatasetManager:
    def __init__(self):
        self.datasets_dir = Path('datasets')
        self.datasets_dir.mkdir(exist_ok=True)
        
    def download_sample_dataset(self):
        """Download a sample dataset for training"""
        print("Setting up sample dataset structure...")
        
        # Create YOLO format directories
        sample_dir = self.datasets_dir / 'sample_rescue'
        for split in ['train', 'val']:
            (sample_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (sample_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create dataset config
        config = {
            'path': str(sample_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 2,  # number of classes
            'names': ['person', 'animal']
        }
        
        config_path = sample_dir / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"Dataset structure created at: {sample_dir}")
        print("Add your training images and labels to complete the setup")
        
        return config_path
    
    def create_training_config(self, dataset_path):
        """Create training configuration"""
        config = {
            'model': 'yolov8n.pt',
            'data': str(dataset_path),
            'epochs': 100,
            'patience': 50,
            'batch': 16,
            'imgsz': 640,
            'save_period': 10,
            'device': 'cpu',  # Change to 'cuda' if GPU available
            'workers': 4,
            'project': 'runs/train',
            'name': 'disaster_rescue'
        }
        
        config_path = Path('config/training_config.yaml')
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path

if __name__ == "__main__":
    dm = DatasetManager()
    dataset_config = dm.download_sample_dataset()
    training_config = dm.create_training_config(dataset_config)
    print(f"Training config saved to: {training_config}")