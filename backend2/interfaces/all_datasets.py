from pydantic import BaseModel


class AllDatasetsResponse(BaseModel):
    datasets: list[str]
