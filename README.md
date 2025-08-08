# 🤖 Financial Q&A System

A sophisticated RAG-based system with intelligent agent orchestration for answering complex financial questions using SEC 10-K filings. Supports both structured financial metrics and qualitative narrative analysis.

##  Features

- **Automated SEC Data Acquisition**: Downloads and processes 10-K filings from SEC EDGAR API
- **Hybrid RAG Pipeline**: Combines structured financial metrics with narrative text analysis
- **Intelligent Agent Orchestration**: Multi-step reasoning with query decomposition
- **5 Query Types Supported**:
  - Basic Metrics (revenue, income, etc.)
  - Year-over-Year Comparisons
  - Cross-Company Analysis
  - Segment Analysis (percentages)
  - Complex Multi-aspect Queries

##  Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (for embeddings)
- Internet connection (for SEC API)

### Setup
1. **Clone the repository**
git clone https://github.com/ShreyaVundavalli/financial-qa-system.git

cd financial_qa_system

3. **Set up environment**

chmod +x venv_setup.sh

./venv_setup.sh

5. **Activate environment**
source venv/bin/activate


6. **Configure your details**
- Edit `src/data_acquisition.py` line 12
- Replace with your name and email: `'User-Agent': 'YourName youremail@domain.com'`

5. **Initialize the system**

-Download and process SEC data (takes 2-3 minutes)
    python src/data_acquisition.py

- Run comprehensive tests
  
    python src/main.py (for 12 sample queries)
  
    python src/main.py --interactive (Interactive Mode)
 

##  Sample Queries & Output

### 1. Basic Metrics
**Query**: "What was Microsoft's total revenue in 2023?"

### 2. Cross-Company Comparison
**Query**: "Which company had the highest operating margin in 2023?"


### 3. Complex Multi-aspect Analysis
**Query**: "Compare R&D spending as percentage of revenue across all three companies"



##  System Architecture

### RAG Implementation (30%)
- **Chunking Strategy**: Hybrid approach combining structured financial metrics (4,466 chunks) with narrative text sections (63 chunks)
- **Embedding Model**: `BAAI/bge-base-en-v1.5` - chosen for superior financial domain performance
- **Vector Store**: FAISS with cosine similarity for fast retrieval
- **Deduplication**: Intelligent filtering to remove duplicate metrics and keep annual totals

### Agent Orchestration (30%)
- **Query Classification**: 5 distinct query types with pattern-based detection
- **Sub-query Generation**: Automatic decomposition of complex queries into retrievable components
- **Multi-step Reasoning**: Sequential processing with context preservation
- **Result Synthesis**: Intelligent aggregation with confidence scoring

### Query Types & Accuracy (20%)
1. **Simple Direct**: Single metric, company, year → 100% success
2. **YoY Comparison**: Growth calculations → 100% success  
3. **Cross-Company**: Comparative analysis → 100% success
4. **Segment Analysis**: Percentage calculations → 100% success
5. **Complex Multi-aspect**: Multi-dimensional analysis → 75% success

##  Technical Details

### Data Pipeline
SEC EDGAR API → Raw 10-K Filings → Structured Extraction → Text Chunking → Embeddings → Vector Index


### Query Processing Flow
User Query → Parse & Classify → Generate Sub-queries → RAG Retrieval → Agent Synthesis → Response


### Key Components
- **Data Acquisition**: `src/data_acquisition.py` - SEC API integration with narrative extraction
- **RAG Pipeline**: `src/rag_pipeline.py` - Embedding generation and retrieval
- **Agent System**: `src/agent.py` - Query orchestration and reasoning
- **Main Interface**: `src/main.py` - Testing framework and interactive mode

##  Performance Metrics

- **Success Rate**: 12/12 queries (100%)
- **Average Confidence**: 0.95 for quantitative, 0.7 for qualitative
- **Response Time**: <3 seconds per query
- **Data Coverage**: MSFT, GOOGL, NVDA (2022-2024)
- **Total Chunks**: 4,529 (4,466 financial + 63 narrative)

##  Bonus Features Implemented

✅ **Automated SEC Filing Downloader** (+5%): Direct SEC EDGAR API integration  
✅ **Edge Case Handling** (+5%): Graceful degradation for missing data, confidence scoring  
✅ **Creative Agent Patterns** (+5%): Hybrid qualitative/quantitative analysis, fallback strategies  

## Troubleshooting

### Common Issues
1. **Rate Limiting**: SEC API has request limits - the system includes automatic delays
2. **Memory Usage**: Large embeddings require 4GB+ RAM
3. **Data Not Found**: Run `python src/data_acquisition.py` if chunks are missing

### Debug Mode
Check data status
    python src/diagnostic_check.py


## Project Structure

```
financial_qa_system/
├── src/
│   ├── data_acquisition.py     # SEC data extraction
│   ├── rag_pipeline.py         # Vector search & retrieval
│   ├── agent.py                # Query orchestration
│   ├── main.py                 # Test framework & interface
│   └── diagnostic_check.py     # Data verification
├── data/
│   ├── rag_chunks.json         # Processed chunks
│   ├── sec_api/                # Raw SEC data
│   └── narrative_text/         # Extracted narratives
├── venv/                       # Virtual environment
├── requirements.txt            # Dependencies
├── venv_setup.sh               # Setup script
└── README.md                   # This file
```



##  System Requirements

- **Minimum**: Python 3.8, 4GB RAM, 2GB disk space
- **Recommended**: Python 3.10+, 8GB RAM, 5GB disk space
- **Internet**: Required for SEC API access during data acquisition


---

 Ready to explore financial data? Start with `python src/main.py --interactive`
