from typing import Dict, List, Optional, Any

from sqlmodel import Field, Column, JSON
from sqlalchemy import Text, UniqueConstraint

from minds.model.base import BaseSQLModel


class Mind(BaseSQLModel, table=True):
    __tablename__ = "minds"

    # Basic mind information
    name: str = Field(description="Name of the mind", max_length=256, index=True)
    provider: str = Field(description="AI provider (openai, google, etc.)", max_length=50)
    model_name: str = Field(description="Model name to use", max_length=256)
    
    # User/Company ownership
    user_id: str = Field(description="ID of the user who owns this mind", max_length=256, index=True)
    company_id: str = Field(description="ID of the company this mind belongs to", max_length=256, index=True)
    
    # Configuration
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Mind parameters and configuration as JSON"
    )
    
    # Associated resources (stored as JSON arrays)
    datasources: Optional[List[str]] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="List of datasource names attached to this mind"
    )
    
    # Optional metadata
    description: Optional[str] = Field(
        default=None,
        sa_column=Column(Text),
        description="Optional description of the mind"
    )
    
    # Status tracking
    is_active: bool = Field(default=True, description="Whether the mind is active")
    
    # Database constraints
    __table_args__ = (
        UniqueConstraint('name', 'company_id', name='unique_mind_name_per_company'),
    )
    
    def add_datasource(self, datasource_name: str) -> None:
        """Add a datasource to this mind."""
        if self.datasources is None:
            self.datasources = []
        if datasource_name not in self.datasources:
            self.datasources.append(datasource_name)
    
    def remove_datasource(self, datasource_name: str) -> bool:
        """Remove a datasource from this mind. Returns True if removed, False if not found."""
        if self.datasources and datasource_name in self.datasources:
            self.datasources.remove(datasource_name)
            return True
        return False

