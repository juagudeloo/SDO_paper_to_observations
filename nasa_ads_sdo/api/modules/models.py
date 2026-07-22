from sqlmodel import Field, SQLModel
from typing import Optional

class SDODocumentBase(SQLModel):
    title: str
    abstract: str
    authors: str
    publication_date: str = Field(index=True)
    doi: str | None = None
    bibcode: str | None = None
    citation_count: int | None = None
    
class SDODocument(SDODocumentBase, table=True):
    id: int = Field(default=None, primary_key=True)

class SDODocumentPublic(SDODocumentBase):
    id: int
    ads_url: Optional[str] = None