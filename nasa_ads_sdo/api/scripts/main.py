from fastapi import FastAPI, Depends, HTTPException, Query, Path
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select
from typing import List, Optional
import sys
import os
from pathlib import Path as PathLib
import httpx
import io

# Add the parent directory to the path to import modules
current_dir = PathLib(__file__).parent
api_dir = current_dir.parent
sys.path.insert(0, str(api_dir))

from modules.database import engine
from modules.models import SDODocument, SDODocumentPublic
from modules.config import API_TITLE, API_DESCRIPTION, API_VERSION, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

def get_session():
    with Session(engine) as session:
        yield session

@app.get("/documents/", response_model=List[SDODocumentPublic])
def read_documents(
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Number of documents to return"),
    year: Optional[int] = Query(None, description="Filter by publication year"),
    session: Session = Depends(get_session)
):
    """Get a list of SDO documents with optional filtering and pagination."""
    query = select(SDODocument)
    
    if year:
        query = query.where(SDODocument.publication_date.like(f"{year}%"))
    
    query = query.offset(skip).limit(limit)
    documents = session.exec(query).all()
    
    # Convert to public model and add ADS URLs
    public_documents = []
    for doc in documents:
        doc_data = {
            "id": doc.id,
            "title": doc.title,
            "abstract": doc.abstract,
            "authors": doc.authors,
            "publication_date": doc.publication_date,
            "doi": doc.doi,
            "bibcode": doc.bibcode,
            "citation_count": doc.citation_count,
            "ads_url": f"https://ui.adsabs.harvard.edu/abs/{doc.bibcode}" if doc.bibcode else None
        }
        public_documents.append(SDODocumentPublic(**doc_data))
    
    return public_documents

@app.get("/documents/{document_id}", response_model=SDODocumentPublic)
def read_document(document_id: int, session: Session = Depends(get_session)):
    """Get a specific SDO document by ID."""
    document = session.get(SDODocument, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Convert to public model and add ADS URL
    doc_data = {
        "id": document.id,
        "title": document.title,
        "abstract": document.abstract,
        "authors": document.authors,
        "publication_date": document.publication_date,
        "doi": document.doi,
        "bibcode": document.bibcode,
        "citation_count": document.citation_count,
        "ads_url": f"https://ui.adsabs.harvard.edu/abs/{document.bibcode}" if document.bibcode else None
    }
    
    return SDODocumentPublic(**doc_data)

@app.get("/documents/search/", response_model=List[SDODocumentPublic])
def search_documents(
    q: str = Query(..., description="Search query for title or abstract"),
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    session: Session = Depends(get_session)
):
    """Search documents by title or abstract content."""
    query = select(SDODocument).where(
        (SDODocument.title.contains(q)) | (SDODocument.abstract.contains(q))
    ).offset(skip).limit(limit)
    
    documents = session.exec(query).all()
    
    # Convert to public model and add ADS URLs
    public_documents = []
    for doc in documents:
        doc_data = {
            "id": doc.id,
            "title": doc.title,
            "abstract": doc.abstract,
            "authors": doc.authors,
            "publication_date": doc.publication_date,
            "doi": doc.doi,
            "bibcode": doc.bibcode,
            "citation_count": doc.citation_count,
            "ads_url": f"https://ui.adsabs.harvard.edu/abs/{doc.bibcode}" if doc.bibcode else None
        }
        public_documents.append(SDODocumentPublic(**doc_data))
    
    return public_documents

@app.get("/documents/{document_id}/download-pdf")
async def download_pdf_auto(
    document_id: int, 
    source: str = Query(None, description="Preferred PDF source: 'arxiv' or 'publisher'. If not specified, tries arXiv first, then publisher."),
    session: Session = Depends(get_session)
):
    """Download PDF for a specific document, trying arXiv first, then publisher (or specific source if requested)."""
    document = session.get(SDODocument, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not document.bibcode:
        raise HTTPException(status_code=404, detail="Document does not have a bibcode")
    
    # Determine sources to try based on the source parameter
    if source:
        if source not in ["arxiv", "publisher"]:
            raise HTTPException(status_code=400, detail="Source must be 'arxiv' or 'publisher'")
        sources = [(source, f"https://ui.adsabs.harvard.edu/link_gateway/{document.bibcode}/{'EPRINT_PDF' if source == 'arxiv' else 'PUB_PDF'}")]
    else:
        # Try arXiv first, then publisher
        sources = [
            ("arxiv", f"https://ui.adsabs.harvard.edu/link_gateway/{document.bibcode}/EPRINT_PDF"),
            ("publisher", f"https://ui.adsabs.harvard.edu/link_gateway/{document.bibcode}/PUB_PDF")
        ]
    
    # Set up browser-like headers to avoid blocking
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/pdf,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
    }
    
    async with httpx.AsyncClient(
        follow_redirects=True, 
        timeout=30.0,
        headers=headers,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
    ) as client:
        for source_type, ads_url in sources:
            try:
                response = await client.get(ads_url)
                
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "")
                    
                    # Better PDF detection - avoid HTML error pages
                    is_pdf = (
                        "pdf" in content_type.lower() or 
                        (len(response.content) > 10000 and not (
                            "html" in content_type.lower() or 
                            b"<html" in response.content[:1000].lower()
                        ))
                    )
                    
                    if is_pdf:
                        filename = f"{document.bibcode}_{source_type}.pdf"
                        
                        return StreamingResponse(
                            io.BytesIO(response.content),
                            media_type="application/pdf",
                            headers={
                                "Content-Disposition": f"attachment; filename={filename}",
                                "Content-Length": str(len(response.content)),
                                "X-PDF-Source": source_type,
                                "X-Original-URL": str(response.url)  # Show the final URL after redirects
                            }
                        )
                        
            except (httpx.TimeoutException, httpx.RequestError):
                continue  # Try the next source
    
    # If we get here, no PDF was found from either source
    raise HTTPException(
        status_code=404, 
        detail="PDF not available from either arXiv or publisher sources"
    )

@app.get("/documents/{document_id}/download-pdf/{pdf_type}")
async def download_pdf(
    document_id: int, 
    pdf_type: str = Path(..., pattern="^(arxiv|publisher)$", description="PDF type: 'arxiv' or 'publisher'"),
    session: Session = Depends(get_session)
):
    """Download PDF for a specific document from arXiv or publisher via NASA ADS."""
    document = session.get(SDODocument, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not document.bibcode:
        raise HTTPException(status_code=404, detail="Document does not have a bibcode")
    
    # Determine the correct ADS link gateway endpoint
    if pdf_type == "arxiv":
        ads_url = f"https://ui.adsabs.harvard.edu/link_gateway/{document.bibcode}/EPRINT_PDF"
        filename = f"{document.bibcode}_arxiv.pdf"
    else:  # publisher
        ads_url = f"https://ui.adsabs.harvard.edu/link_gateway/{document.bibcode}/PUB_PDF"
        filename = f"{document.bibcode}_publisher.pdf"
    
    try:
        # Set up browser-like headers to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/pdf,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
        
        async with httpx.AsyncClient(
            follow_redirects=True, 
            timeout=30.0,
            headers=headers,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        ) as client:
            # First, follow the NASA ADS redirect to get the actual PDF URL
            response = await client.get(ads_url)
            
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404, 
                    detail=f"PDF not available from {pdf_type} source"
                )
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=502, 
                    detail=f"Failed to retrieve PDF from NASA ADS (status: {response.status_code})"
                )
            
            # Check if the response is actually a PDF
            content_type = response.headers.get("content-type", "")
            
            # If we got redirected to a login or error page, the content type might not be PDF
            if "pdf" not in content_type.lower():
                # Try to detect if this is an HTML error page
                if "html" in content_type.lower() or (
                    len(response.content) < 10000 and 
                    b"<html" in response.content[:1000].lower()
                ):
                    raise HTTPException(
                        status_code=404, 
                        detail=f"PDF not available from {pdf_type} source (redirected to non-PDF content)"
                    )
            
            # For small responses that aren't clearly PDFs, check content
            if len(response.content) < 1000 and "pdf" not in content_type.lower():
                raise HTTPException(
                    status_code=404, 
                    detail=f"PDF not available from {pdf_type} source (invalid content)"
                )
            
            # Create a streaming response with the PDF content
            return StreamingResponse(
                io.BytesIO(response.content),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Content-Length": str(len(response.content)),
                    "X-PDF-Source": pdf_type,
                    "X-Original-URL": str(response.url)  # Show the final URL after redirects
                }
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout while fetching PDF from NASA ADS")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to NASA ADS: {str(e)}")

@app.get("/documents/{document_id}/ads-links")
def get_ads_links(document_id: int, session: Session = Depends(get_session)):
    """Get ADS-related links for a specific document."""
    document = session.get(SDODocument, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not document.bibcode:
        raise HTTPException(status_code=404, detail="Document does not have a bibcode")
    
    # Get the base URL from the request (for API download links)
    base_url = "http://localhost:8000"  # You can make this dynamic if needed
    
    ads_links = {
        "bibcode": document.bibcode,
        "ads_url": f"https://ui.adsabs.harvard.edu/abs/{document.bibcode}",
        "pdf_links": {
            "arxiv_pdf_direct": f"https://ui.adsabs.harvard.edu/link_gateway/{document.bibcode}/EPRINT_PDF",
            "publisher_pdf_direct": f"https://ui.adsabs.harvard.edu/link_gateway/{document.bibcode}/PUB_PDF"
        },
        "api_download_links": {
            "download_pdf_auto": f"{base_url}/documents/{document_id}/download-pdf",
            "download_arxiv_pdf": f"{base_url}/documents/{document_id}/download-pdf/arxiv",
            "download_publisher_pdf": f"{base_url}/documents/{document_id}/download-pdf/publisher"
        },
        "export_links": {
            "bibtex": f"https://ui.adsabs.harvard.edu/abs/{document.bibcode}/exportcitation",
            "ads_format": f"https://ui.adsabs.harvard.edu/abs/{document.bibcode}/exportcitation"
        },
        "related_links": {
            "references": f"https://ui.adsabs.harvard.edu/abs/{document.bibcode}/references",
            "citations": f"https://ui.adsabs.harvard.edu/abs/{document.bibcode}/citations",
            "similar": f"https://ui.adsabs.harvard.edu/abs/{document.bibcode}/similar"
        }
    }
    
    return ads_links

@app.get("/stats/")
def get_stats(session: Session = Depends(get_session)):
    """Get basic statistics about the document collection."""
    total_docs = session.exec(select(SDODocument)).all()
    total_count = len(total_docs)
    
    years = [int(doc.publication_date[:4]) for doc in total_docs if doc.publication_date and len(doc.publication_date) >= 4]
    year_range = {"min": min(years), "max": max(years)} if years else None
    
    return {
        "total_documents": total_count,
        "year_range": year_range
    }

@app.get("/")
def root():
    """API root endpoint."""
    return {"message": API_TITLE, "version": API_VERSION, "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    from modules.config import HOST, PORT, DEBUG
    uvicorn.run(app, host=HOST, port=PORT, reload=DEBUG)
