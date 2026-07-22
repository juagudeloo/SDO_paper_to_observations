import pandas as pd
import requests
import os
from urllib.parse import urlencode
from sqlmodel import Session
from dotenv import load_dotenv
load_dotenv("../.env")

import sys
sys.path.append("../")
from modules.database import create_db_and_tables, engine
from modules.models import SDODocument
from sqlmodel import SQLModel

token = os.getenv("NASA_ADS_API_KEY")

search_url = "https://api.adsabs.harvard.edu/v1/search/query"


def main():
    # Drop existing tables and recreate with new schema
    SQLModel.metadata.drop_all(engine)
    create_db_and_tables()
    
    # Iterate through all years from 2010 to 2024
    for year in range(2010, 2025):
        print(f"Extracting documents for year {year}...")
        docs = extract_sdo_documents(year)
        print(f"Found {len(docs)} documents for year {year}")
        load_sdo_documents(docs)
        print(f"Loaded documents for year {year} into database")

def extract_sdo_documents(pub_year):
    # Manually construct field list, mapping publication_date to pubdate for API
    api_fields = []
    for col in SDODocument.__table__.columns:
        if col.name == 'publication_date':
            api_fields.append('pubdate')
        else:
            api_fields.append(col.name)
    
    fl_fields = ",".join(api_fields)

    encoded_url = urlencode({
        "q": f"abstract:SDO, year:{pub_year}, body:'solar dynamics observatory'",
        "fq": "property:refereed",
        "sort": "date desc",
        "fl": fl_fields,
        "rows": 2000
    })
    
    results = requests.get(f"{search_url}?{encoded_url}", 
                           headers={"Authorization": f"Bearer {token}"}).json()

    return results['response']['docs']

def load_sdo_documents(docs):
    with Session(engine) as session:
        for doc in docs:
            sdo_doc = SDODocument(
                id=int(doc.get('id')),
                title=doc.get('title', [''])[0],
                abstract=doc.get('abstract', ''),
                authors=", ".join(doc.get('author', [])),
                publication_date=str(doc.get('pubdate', 0)),
                doi=doc.get('doi', [None])[0],
                bibcode=doc.get('bibcode', None),
                citation_count=doc.get('citation_count', None)
            )
            session.add(sdo_doc)
        session.commit()
        
if __name__ == "__main__":
    main()