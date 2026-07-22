# NASA- 🚀 **FastAPI Web Service**: RESTful API for accessing SDO research documents
- 🔗 **NASA ADS Integration**: Automatic generation of ADS URLs### Get comprehensive ADS links for a document
```bash
curl "http://localhost:8000/documents/1366704/ads-links"
```

### Download PDF (automatic source selection)
```bash
# Downloads PDF if available (tries arXiv first, then publisher)
# Uses browser-like headers and follows redirects automatically
curl -o paper.pdf "http://localhost:8000/documents/1366704/download-pdf"
```

### Download PDF from specific source
```bash
# Download from arXiv
curl -o paper_arxiv.pdf "http://localhost:8000/documents/1366704/download-pdf?source=arxiv"

# Download from publisher (most reliable for this document)
curl -o paper_publisher.pdf "http://localhost:8000/documents/1366704/download-pdf?source=publisher"
```

### Get API statistics
```bash
curl "http://localhost:8000/stats/"
``` document
- 📄 **PDF Download**: Direct PDF download endpoints via NASA ADS link gateway
- 🔍 **Search Functionality**: Search documents by title, abstract, or publication year SDO Database & API

Based on the [NASA ADS API](https://ui.adsabs.harvard.edu/help/api/), this repository contains the necessary code to obtain all the published papers that make use of SDO (Solar Dynamics Observatory) data for solar atmospheric analysis, along with a FastAPI web service to access this data.

## Features

- 🚀 **FastAPI Web Service**: RESTful API for accessing SDO research documents
- � **NASA ADS Integration**: Automatic generation of ADS URLs for each document
- 📄 **PDF Access**: Direct links to arXiv and publisher PDFs via ADS
- �🔍 **Search Functionality**: Search documents by title, abstract, or publication year
- 📊 **Statistics**: Get insights about the document collection
- 📖 **Interactive Documentation**: Automatic API documentation with Swagger UI
- 🐍 **Python 3.8+**: Modern Python support with type hints
- 💾 **SQLite Database**: Lightweight, file-based database for easy deployment

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

**On Ubuntu/Debian systems, you also need:**
```bash
sudo apt update
sudo apt install python3-venv
```

**On other systems, ensure you have:**
- `python3-venv` or equivalent virtual environment support

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/juagudeloo/NASA_ADS_SDO.git
   cd NASA_ADS_SDO
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
   
   Or using Make:
   ```bash
   make install
   ```

   This script will:
   - Create a Python virtual environment
   - Install all required dependencies
   - Create executable scripts for running the API

### Running the API

#### Option 1: Using the run scripts

**Production mode:**
```bash
./run_api.sh
```

**Development mode (with auto-reload):**
```bash
./run_dev.sh
```

#### Option 2: Using Make commands

**Production mode:**
```bash
make run
```

**Development mode:**
```bash
make dev
```

#### Option 3: Using Docker

**Build and run with Docker:**
```bash
docker build -t nasa-ads-sdo-api .
docker run -p 8000:8000 -v $(pwd)/api/database:/app/api/database:ro nasa-ads-sdo-api
```

#### Option 4: Manual execution

```bash
# Activate virtual environment
source venv/bin/activate

# Navigate to scripts directory
cd api/scripts

# Run the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the API

Once the server is running, you can access:

- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

## API Endpoints

### Documents

- `GET /documents/` - Get paginated list of documents with ADS URLs
  - Query parameters: `skip`, `limit`, `year`
  - **New**: Each document includes `ads_url` field pointing to NASA ADS
- `GET /documents/{id}` - Get specific document by ID with ADS URL
- `GET /documents/{id}/ads-links` - Get comprehensive ADS-related links for a document
  - Returns: ADS URL, PDF links (arXiv/publisher), export links, related links
- `GET /documents/{id}/download-pdf` - Download PDF automatically (tries arXiv first, then publisher)
- `GET /documents/{id}/download-pdf?source=arxiv` - Download PDF from arXiv
- `GET /documents/{id}/download-pdf?source=publisher` - Download PDF from publisher
- `GET /documents/search/` - Search documents by title or abstract with ADS URLs
  - Query parameters: `q` (search query), `skip`, `limit`

### Statistics

- `GET /stats/` - Get collection statistics (total documents, year range)

### System

- `GET /` - API information and health check

## Examples

### Get documents from 2020 with ADS URLs
```bash
curl "http://localhost:8000/documents/?year=2020&limit=10"
```

### Search for documents about "coronal mass ejections"
```bash
curl "http://localhost:8000/documents/search/?q=coronal%20mass%20ejections&limit=5"
```

### Get comprehensive ADS links for a specific document
```bash
curl "http://localhost:8000/documents/1366704/ads-links"
```

### Get API statistics
```bash
curl "http://localhost:8000/stats/"
```

### Example Response with ADS URL
```json
{
  "id": 1366704,
  "title": "The EUV spectrum of the Sun: long-term variations...",
  "abstract": "We present SOHO Coronal Diagnostic Spectrometer...",
  "authors": "Del Zanna, G.; Andretta, V.; ...",
  "publication_date": "2010-07-00",
  "doi": "10.1051/0004-6361/200912904",
  "bibcode": "2010A&A...518A..49D",
  "citation_count": 27,
  "ads_url": "https://ui.adsabs.harvard.edu/abs/2010A&A...518A..49D"
}
```

## Development

### Available Make Commands

```bash
make help       # Show available commands
make install    # Install dependencies and set up environment
make run        # Run API in production mode
make dev        # Run API in development mode
make clean      # Clean up virtual environment and cache files
make check-db   # Check if database exists and show info
```

### Project Structure

```
NASA_ADS_SDO/
├── api/
│   ├── database/
│   │   └── sdo_papers_2010_2024.db    # SQLite database
│   ├── modules/
│   │   ├── config.py                  # Configuration settings
│   │   ├── database.py                # Database connection
│   │   └── models.py                  # Data models
│   └── scripts/
│       └── main.py                    # FastAPI application
├── requirements.txt                   # Python dependencies
├── setup.sh                          # Setup script
├── run_api.sh                        # Production runner
├── run_dev.sh                        # Development runner
├── Dockerfile                        # Docker configuration
├── Makefile                          # Make commands
└── README.md                         # This file
```

### Configuration

The API can be configured using environment variables in the `.env` file:

```properties
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# NASA ADS API Key (for future data collection)
NASA_ADS_API_KEY=your_api_key_here

# Database URL (optional, defaults to SQLite)
DATABASE_URL=sqlite:///api/database/sdo_papers_2010_2024.db
```

## Database

The project uses a SQLite database containing SDO research papers from 2010-2024. The database includes:

- **Title**: Paper title
- **Abstract**: Paper abstract
- **Authors**: List of authors
- **Publication Date**: When the paper was published
- **DOI**: Digital Object Identifier
- **Bibcode**: NASA ADS bibliographic code (used to generate ADS URLs)
- **Citation Count**: Number of citations
- **ADS URL**: Automatically generated link to the paper on NASA ADS (computed from bibcode)

### Database Schema

```python
class SDODocument:
    id: int                    # Primary key
    title: str                # Paper title
    abstract: str             # Paper abstract
    authors: str              # Authors list
    publication_date: str     # Publication date
    doi: str | None          # DOI (optional)
    bibcode: str | None      # ADS bibcode (optional)
    citation_count: int | None # Citation count (optional)
```

## Troubleshooting

### Common Issues

1. **Virtual environment creation fails (Ubuntu/Debian)**
   ```
   Error: ensurepip is not available
   ```
   **Solution:**
   ```bash
   sudo apt update
   sudo apt install python3-venv
   ```

2. **Database not found error**
   ```bash
   make check-db  # Check if database exists
   ```

3. **Permission denied on scripts**
   ```bash
   chmod +x setup.sh run_api.sh run_dev.sh
   ```

4. **Python version issues**
   - Ensure Python 3.8+ is installed
   - Check with: `python3 --version`

5. **Port already in use**
   - Change the port in `.env` file or kill the process using port 8000

6. **PDF download returns 404 error**
   ```
   HTTP 404: PDF not available for this document
   ```
   **Explanation:**
   - Not all papers have publicly available PDFs through NASA ADS
   - Some papers may only be available through subscription-based publishers
   - Try different source options: `?source=arxiv` or `?source=publisher`
   - Check the `/ads-links` endpoint first to see available download links
   - The API now uses browser-like headers and follows redirects to improve success rate

7. **PDF download is slow or times out**
   - PDFs are streamed directly from NASA ADS link gateway after following redirects
   - Large papers may take time to download (the API supports up to 30-second timeout)
   - Network connectivity to NASA ADS servers and publisher websites affects download speed
   - The API includes download progress headers showing the original source URL

### Clean Installation

If you encounter issues, try a clean installation:

```bash
make clean      # Remove virtual environment and cache
make install    # Reinstall everything
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the changes
5. Submit a pull request

## Production Deployment

### Systemd Service (Linux)

1. Copy the project to `/opt/nasa-ads-sdo-api`
2. Run the setup script
3. Copy the service file:
   ```bash
   sudo cp nasa-ads-sdo-api.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable nasa-ads-sdo-api
   sudo systemctl start nasa-ads-sdo-api
   ```

### Docker Deployment

```bash
# Build the image
docker build -t nasa-ads-sdo-api .

# Run the container
docker run -p 8000:8000 -v $(pwd)/api/database:/app/api/database:ro nasa-ads-sdo-api

# Check logs
docker logs <container-id>
```

### Reverse Proxy (Nginx)

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## License

This project is open source. Please check the repository for license details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the API documentation at `/docs` when running the server
- Review the troubleshooting section above
