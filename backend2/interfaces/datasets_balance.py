from pydantic import BaseModel


class DatasetBalanceResponse(BaseModel):
    balance: dict[str, int]
