from sqlmodel import Field

from minds.model.base_sql_model import BaseSQLModel


class Mind(BaseSQLModel, table=True):
    __tablename__ = "minds"

    name: str = Field(description="Name of the mind")
