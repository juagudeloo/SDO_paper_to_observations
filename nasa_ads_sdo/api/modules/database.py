from sqlmodel import SQLModel, create_engine
import os
from pathlib import Path

# Get the absolute path to the database file
current_dir = Path(__file__).parent
database_dir = current_dir.parent / "database"
sqlite_file_path = database_dir / "sdo_papers_2010_2024.db"
sqlite_url = f"sqlite:///{sqlite_file_path}"

engine = create_engine(sqlite_url)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)