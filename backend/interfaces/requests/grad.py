from pydantic import BaseModel

class GradRequest(BaseModel):
    model_name: str
    dataset_name: str
    label_name: str
    filename: str
    target_label: int
    normalize: bool