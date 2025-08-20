import os
from torchvision.datasets import ImageFolder
from dataclasses import dataclass
import base64


@dataclass
class DatasetInfo:
    name: str
    dataset: ImageFolder
    path: str

class DatasetsService:

    def __init__(self, datasets_path: str):
        self.datasets_path = datasets_path

    def list_available_datasets(self) -> list[str]:
        datasets = os.listdir(self.datasets_path)
        return datasets
    
    def list_labels_in_dataset(self, dataset_name: str) -> list[str]:
        path = os.path.join(self.datasets_path, dataset_name)
        labels = os.listdir(path)
        return labels
        

    def list_filenames_in_label(self, dataset_name: str, label: str) -> list[str]:
        path = os.path.join(self.datasets_path, dataset_name, label)
        filenames = os.listdir(path)
        return filenames
    
    def load_dataset(self, dataset_name: str, transform=None) -> DatasetInfo:
        path = os.path.join(self.datasets_path, dataset_name)

        dataset = ImageFolder(path, transform=transform)

        dataset_info = DatasetInfo(
            name=dataset_name,
            dataset=dataset,
            path=path
        )

        return dataset_info
    
    def fetch_image(self,
                    dataset: str,
                    label: str,
                    filename: str) -> bytes:
        path = os.path.join(
            self.datasets_path,
            dataset,
            label,
            filename
        )

        with open(path, 'rb') as f:
            image = base64.b64encode(f.read()).decode('utf-8')

        return image