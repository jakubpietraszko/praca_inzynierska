from pydantic import BaseModel

class LabelFilename(BaseModel):
    label: str
    filename: str

class LabelFilenameTarget(LabelFilename):
    target: int