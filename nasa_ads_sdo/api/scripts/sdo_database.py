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

token = os.getenv("SCIX_API_KEY")

# NASA ADS's classic search API is superseded by SciX (scixplorer.org), the
# platform intended to replace it; same query syntax (Solr-based) and
# Bearer-token auth scheme, just a different host.
search_url = "https://scixplorer.org/v1/search/query"


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
        # "Is this genuinely an SDO paper" is an OR of two signals, so either
        # is sufficient: (1) bibgroup:SDO — ADS's own librarian-curated
        # bibliography for the SDO mission; (2) the FULL phrase "solar dynamics
        # observatory" explicitly named in the ABSTRACT specifically (not body
        # — see below) — not the loose acronym "SDO" alone, which also matches
        # papers that only cite SDO in passing, or unrelated fields where the
        # acronym collides.
        #
        # Two things verified live against the real API on 2026-07-27, each
        # correcting an assumption that silently didn't hold:
        #
        # 1. bibgroup:SDO returns ZERO results — unlike bibgroup:HST, which
        #    works exactly as ADS's own docs describe, so this specific
        #    bibgroup is unpopulated/inactive in the live index, not a syntax
        #    problem. Kept anyway (costs nothing in an OR; would start
        #    contributing for free if ADS/SciX ever populates it), but the
        #    phrase match is what's actually carrying this filter today.
        #
        # 2. Phrase quoting MUST use double quotes (abstract:"...") — ADS's
        #    documented "Advanced Search Syntax" uses double quotes for phrase
        #    search; single quotes (the form originally used here) do NOT
        #    enforce a phrase match and instead let each word match
        #    independently (with synonym expansion — e.g. "solar"~"sun").
        #    Confirmed by a concrete false positive: with single quotes, a
        #    paper on Betelgeuse convection matched even though its abstract
        #    never contains the phrase "solar dynamics observatory" at all —
        #    it separately mentions "sun" (synonym of "solar") and "the
        #    Observatoire du Pic du Midi" (loosely matching "observatory").
        #
        # ALSO verified: requiring the phrase in `body` (full text) instead of
        # just `abstract` roughly triples matches per year and reintroduces
        # clear false positives at the top (e.g. papers on Be-star mass loss,
        # hypervelocity stars) — full-text citations of SDO as a comparison
        # point are common and don't mean the paper is about SDO. Deliberately
        # NOT included; abstract-only is the precise signal.
        #
        # Everything else is a mandatory (AND) quality/domain filter, regardless
        # of which of the above two signals qualified the paper:
        # - bibstem:A&A       — Astronomy & Astrophysics journal: open access
        #                       (reliable PDF downloads) and empirically the
        #                       best signal-to-noise source for this pipeline
        #                       so far. Widen to other peer-reviewed solar
        #                       physics journals later if A&A alone doesn't
        #                       give enough coverage.
        # - database:astronomy — excludes physics/other ADS databases outright,
        #                       rather than relying on text alone.
        # - doctype:article   — excludes meeting abstracts, errata, conference
        #                       proceedings, press releases, etc.
        # - property:refereed (below, in fq) — peer-reviewed only.
        "q": (
            '(bibgroup:SDO OR abstract:"solar dynamics observatory"), '
            f"year:{pub_year}, bibstem:A&A, database:astronomy, doctype:article"
        ),
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