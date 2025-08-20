import os
from typing import Optional
from torchvision.datasets import ImageFolder
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    name: str
    dataset: ImageFolder
    path: str


class DatasetService:
    
    def __init__(self, datasets_path: str):
        self.datasets_path = datasets_path
        self._current_dataset: Optional[DatasetInfo] = None
    
    def load_dataset(self, dataset_name: str, transform=None) -> DatasetInfo:
        path = os.path.join(self.datasets_path, dataset_name)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")
        
        try:
            dataset = ImageFolder(path, transform=transform)
            
            dataset_info = DatasetInfo(
                name=dataset_name,
                dataset=dataset,
                path=path
            )
            
            return dataset_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {str(e)}")
    
    def set_current_dataset(self, dataset_info: DatasetInfo) -> None:
        self._current_dataset = dataset_info
    
    def get_current_dataset(self) -> Optional[DatasetInfo]:
        return self._current_dataset
    
    def list_available_datasets(self) -> list[str]:
        if not os.path.exists(self.datasets_path):
            return []
        
        datasets = []
        for item in os.listdir(self.datasets_path):
            dataset_path = os.path.join(self.datasets_path, item)
            if os.path.isdir(dataset_path):
                datasets.append(item)
        
        return datasets
    
    def get_dataset_labels(self, dataset_name: str) -> list[str]:
        path = os.path.join(self.datasets_path, dataset_name)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")
        
        labels = [label for label in os.listdir(path) if os.path.isdir(os.path.join(path, label))]
        return labels
    
    def get_images_in_label(self, dataset_name: str, label: str) -> list[str]:
        path = os.path.join(self.datasets_path, dataset_name, label)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Label path not found: {path}")
        
        images = [file for file in os.listdir(path) if file.endswith(('.png', '.jpg', '.jpeg'))]
        return images
