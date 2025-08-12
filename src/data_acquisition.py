"""
IMPORTANT: Before running, update the User-Agent header on line 12 with your name and email.
The SEC requires this for API access. Example: 'John Doe john.doe@university.edu'
"""


import requests
import pandas as pd
import json
import os
import re
from typing import Dict, List
from bs4 import BeautifulSoup
import time

class SECAPIDataExtractor:
    def __init__(self):
        
        self.headers = {
            'User-Agent': 'Yourname yourmail@domain.com'  
        }
        self.base_url = "https://data.sec.gov/api/xbrl"
        self.edgar_base = "https://www.sec.gov/Archives/edgar/data"
        
    def get_company_facts(self, ticker_to_cik: Dict[str, str]) -> Dict:
        """Get all financial facts for companies using SEC's Company Facts API"""
        all_company_data = {}
        
        for ticker, cik in ticker_to_cik.items():
            try:
                cik_padded = cik.zfill(10)
                url = f"{self.base_url}/companyfacts/CIK{cik_padded}.json"
                
                print(f"📥 Fetching {ticker} structured data from SEC API...")
                response = requests.get(url, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    all_company_data[ticker] = data
                    print(f"✅ {ticker}: Retrieved {len(data.get('facts', {}))} fact categories")
                else:
                    print(f"❌ {ticker}: Failed to get data (status: {response.status_code})")
                    
            except Exception as e:
                print(f"❌ Error fetching {ticker}: {e}")
        
        return all_company_data

    def get_recent_filings(self, ticker_to_cik: Dict[str, str], years: List[str]) -> Dict:
        """Get recent 10-K filing URLs for narrative text extraction"""
        filing_urls = {}
        
        for ticker, cik in ticker_to_cik.items():
            try:
                cik_padded = cik.zfill(10)
                submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
                
                print(f"📋 Fetching {ticker} filing submissions...")
                response = requests.get(submissions_url, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    filings = data.get('filings', {}).get('recent', {})
                    
                    forms = filings.get('form', [])
                    dates = filings.get('filingDate', [])
                    accession_numbers = filings.get('accessionNumber', [])
                    
                    # Find 10-K filings for target years
                    filing_urls[ticker] = []
                    
                    for i, form in enumerate(forms):
                        if form == '10-K' and i < len(dates) and i < len(accession_numbers):
                            filing_year = dates[i][:4]  # Extract year from YYYY-MM-DD
                            
                            if filing_year in years:
                                accession = accession_numbers[i].replace('-', '')
                                # Construct filing URL
                                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{accession_numbers[i]}-index.htm"
                                
                                filing_urls[ticker].append({
                                    'year': filing_year,
                                    'url': filing_url,
                                    'accession': accession_numbers[i]
                                })
                    
                    print(f"✅ {ticker}: Found {len(filing_urls[ticker])} relevant 10-K filings")
                else:
                    print(f"❌ {ticker}: Failed to get filings (status: {response.status_code})")
                    filing_urls[ticker] = []
                    
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error fetching {ticker} filings: {e}")
                filing_urls[ticker] = []
        
        return filing_urls

    def get_direct_filing_urls(self) -> Dict:
        """Direct URLs for known filings to bypass URL construction issues"""
        return {
            ('MSFT', '2024'): 'https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm',
            ('MSFT', '2023'): 'https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm',
            ('MSFT', '2022'): 'https://www.sec.gov/Archives/edgar/data/789019/000156459022026876/msft-10k_20220630.htm',

            ('GOOGL', '2024'): 'https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm',
            ('GOOGL', '2023'): 'https://www.sec.gov/Archives/edgar/data/1652044/000165204424000022/goog-20231231.htm',
            ('GOOGL', '2022'): 'https://www.sec.gov/Archives/edgar/data/1652044/000165204423000016/goog-20221231.htm',

            ('NVDA', '2024'): 'https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm',
            ('NVDA', '2023'): 'https://www.sec.gov/Archives/edgar/data/1045810/000104581023000017/nvda-20230129.htm',
            ('NVDA', '2022'): 'https://www.sec.gov/Archives/edgar/data/1045810/000104581022000036/nvda-20220130.htm'
        }


    def extract_narrative_text(self, ticker: str, cik: str, accession: str, year: str) -> Dict[str, str]:
        """Extract using direct URLs first, then fallback to pattern matching"""
        try:
            # Try direct URL first
            direct_urls = self.get_direct_filing_urls()
            direct_url = direct_urls.get((ticker, year))
            
            if direct_url:
                print(f"  📄 Using direct URL for {ticker} {year}")
                response = requests.get(direct_url, headers=self.headers, timeout=30)
                if response.status_code == 200:
                    return self._parse_10k_sections(response.text, ticker, year)
            
            # Fallback to pattern matching (previous method)
            # ... rest of the enhanced URL pattern logic above
            
        except Exception as e:
            print(f"  ❌ Error extracting narrative from {ticker} {year}: {e}")
            return {}


    def _parse_10k_sections(self, html_content: str, ticker: str, year: str) -> Dict[str, str]:
        """IMPROVED: Parse specific sections from 10-K HTML content with better patterns"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            text = ' '.join(line for line in lines if line)
            
            # IMPROVED: More comprehensive section extraction patterns
            sections = {}
            
            # Risk Factors section - IMPROVED PATTERN
            risk_patterns = [
                r'(ITEM\s*1A[\.\s]*RISK\s*FACTORS.*?)(?=ITEM\s*1B|ITEM\s*2\.|UNRESOLVED\s*STAFF\s*COMMENTS|$)',
                r'(Item\s*1A[\.\s]*Risk\s*Factors.*?)(?=Item\s*1B|Item\s*2\.|$)',
                r'(RISK\s*FACTORS.*?)(?=ITEM\s*1B|ITEM\s*2|UNRESOLVED|$)'
            ]
            
            for pattern in risk_patterns:
                risk_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if risk_match and len(risk_match.group(1)) > 1000:  # Ensure substantial content
                    sections['risk_factors'] = risk_match.group(1)[:15000]  # First 15k chars
                    break
            
            # Business section - IMPROVED PATTERN
            business_patterns = [
                r'(ITEM\s*1[\.\s]*BUSINESS.*?)(?=ITEM\s*1A|ITEM\s*2|$)',
                r'(Item\s*1[\.\s]*Business.*?)(?=Item\s*1A|Item\s*2|$)',
                r'(BUSINESS.*?)(?=RISK\s*FACTORS|ITEM\s*1A|$)'
            ]
            
            for pattern in business_patterns:
                business_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if business_match and len(business_match.group(1)) > 1000:
                    sections['business'] = business_match.group(1)[:12000]
                    break
            
            # MD&A section - IMPROVED PATTERN
            mda_patterns = [
                r'(ITEM\s*7[\.\s]*MANAGEMENT[\'\s]*S\s*DISCUSSION\s*AND\s*ANALYSIS.*?)(?=ITEM\s*7A|ITEM\s*8|$)',
                r'(Item\s*7[\.\s]*Management[\'\s]*s\s*Discussion\s*and\s*Analysis.*?)(?=Item\s*7A|Item\s*8|$)',
                r'(MANAGEMENT[\'\s]*S\s*DISCUSSION\s*AND\s*ANALYSIS.*?)(?=QUANTITATIVE\s*AND\s*QUALITATIVE|ITEM\s*7A|$)'
            ]
            
            for pattern in mda_patterns:
                mda_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if mda_match and len(mda_match.group(1)) > 1000:
                    sections['mda'] = mda_match.group(1)[:12000]
                    break
            
            print(f"    ✅ Extracted {len(sections)} narrative sections from {ticker} {year}")
            print(f"       Risk factors: {len(sections.get('risk_factors', '0'))} chars")
            print(f"       Business: {len(sections.get('business', '0'))} chars")
            print(f"       MD&A: {len(sections.get('mda', '0'))} chars")
            
            return sections
            
        except Exception as e:
            print(f"    ❌ Error parsing sections: {e}")
            return {}


    def save_raw_data(self, company_data: Dict, save_dir: str = "data/sec_api"):
        """Save raw SEC API data to files"""
        os.makedirs(save_dir, exist_ok=True)
        
        for ticker, data in company_data.items():
            file_path = os.path.join(save_dir, f"{ticker}_company_facts.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Saved {ticker} structured data to {file_path}")

    def save_narrative_data(self, narrative_data: Dict, save_dir: str = "data/narrative_text"):
        """Save narrative text data"""
        os.makedirs(save_dir, exist_ok=True)
        
        for ticker, years_data in narrative_data.items():
            for year, sections in years_data.items():
                file_path = os.path.join(save_dir, f"{ticker}_{year}_narrative.json")
                with open(file_path, 'w') as f:
                    json.dump(sections, f, indent=2)
                print(f"💾 Saved {ticker} {year} narrative text to {file_path}")

    def create_text_chunks_for_rag(self, company_data: Dict, narrative_data: Dict, years: List[str]) -> List[Dict]:
        """FIXED: More lenient chunking for narrative text"""
        chunks = []
        
        # Process structured financial data (same as before)
        print("🔢 Processing structured financial data...")
        for ticker, data in company_data.items():
            facts = data.get('facts', {})
            
            for taxonomy_name, taxonomy_data in facts.items():
                for metric_tag, metric_info in taxonomy_data.items():
                    
                    label = metric_info.get('label', 'Unknown Metric')
                    usd_data = metric_info.get('units', {}).get('USD', [])
                    
                    for fact in usd_data:
                        if fact.get('form') == '10-K':
                            year = str(fact.get('fy', ''))
                            value = fact.get('val', 0)
                            
                            if year in years and value is not None:
                                chunk_text = f"{ticker} {year}: {label} was ${value:,} according to their 10-K filing."
                                
                                chunks.append({
                                    'content': chunk_text,
                                    'company': ticker,
                                    'year': year,
                                    'metric': metric_tag,
                                    'label': label,
                                    'value': value,
                                    'source': f"{ticker}_{year}_10K_structured",
                                    'type': 'financial_metric'
                                })
        
        print(f"✅ Created {len(chunks)} financial chunks")
        
        # Process narrative text data - MUCH MORE LENIENT
        print("📄 Processing narrative text data...")
        narrative_chunk_count = 0
        
        for ticker, years_data in narrative_data.items():
            print(f"  Processing {ticker} narrative data...")
            for year, sections in years_data.items():
                print(f"    Processing {ticker} {year} with {len(sections)} sections...")
                for section_name, section_text in sections.items():
                    
                    # MUCH MORE LENIENT: Accept any section with meaningful content
                    if len(section_text.strip()) > 50:  # Reduced from 200 to 50
                        print(f"      Processing {section_name} ({len(section_text)} chars)...")
                        
                        # Clean the section text
                        clean_text = re.sub(r'\s+', ' ', section_text).strip()
                        
                        # VERY SMALL CHUNKS for short sections
                        if len(clean_text) < 500:
                            # For very short sections, create one chunk
                            if len(clean_text) > 50:
                                chunks.append({
                                    'content': clean_text,
                                    'company': ticker,
                                    'year': year,
                                    'metric': section_name,
                                    'label': f"{section_name.replace('_', ' ').title()}",
                                    'value': 0,
                                    'source': f"{ticker}_{year}_10K_{section_name}",
                                    'type': 'narrative_text'
                                })
                                narrative_chunk_count += 1
                                print(f"        ✅ Created 1 chunk from {section_name} (short section)")
                        else:
                            # For longer sections, split into chunks
                            words = clean_text.split()
                            chunk_size = 150  # Smaller chunks for better retrieval
                            overlap = 25
                            
                            section_chunks = 0
                            for i in range(0, len(words), chunk_size - overlap):
                                chunk_words = words[i:i + chunk_size]
                                chunk_text = ' '.join(chunk_words).strip()
                                
                                # VERY LENIENT minimum chunk size
                                if len(chunk_text) > 50:  # Reduced from 100 to 50
                                    chunks.append({
                                        'content': chunk_text,
                                        'company': ticker,
                                        'year': year,
                                        'metric': section_name,
                                        'label': f"{section_name.replace('_', ' ').title()}",
                                        'value': 0,
                                        'source': f"{ticker}_{year}_10K_{section_name}",
                                        'type': 'narrative_text'
                                    })
                                    section_chunks += 1
                                    narrative_chunk_count += 1
                            
                            print(f"        ✅ Created {section_chunks} chunks from {section_name}")
                    else:
                        print(f"        ⚠️  Skipped {section_name} (too short: {len(section_text)} chars)")
        
        print(f"✅ Created {narrative_chunk_count} narrative chunks")
        print(f"📝 Created {len(chunks)} total chunks for RAG pipeline")
        print(f"   - Financial metrics: {len(chunks) - narrative_chunk_count}")
        print(f"   - Narrative text: {narrative_chunk_count}")
        
        return chunks



    def save_chunks_for_rag(self, chunks: List[Dict], save_path: str = "data/rag_chunks.json"):
        """Save chunks in format compatible with RAG pipeline"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(chunks, f, indent=2)
        
        print(f"💾 Saved {len(chunks)} chunks to {save_path}")
        return save_path

def main():
    print("🚀 Starting Enhanced SEC Data Acquisition (Structured + Narrative)\n")
    
    # Company CIK mappings 
    companies = {
        'MSFT': '789019',
        'GOOGL': '1652044', 
        'NVDA': '1045810'
    }
    
    years = ['2022', '2023', '2024']
    
    # Initialize extractor
    extractor = SECAPIDataExtractor()
    
    # Step 1: Get structured financial data from SEC API
    print("Step 1: Fetching structured data from SEC API...")
    company_data = extractor.get_company_facts(companies)
    
    # Step 2: Get 10-K filing URLs for narrative text
    print("\nStep 2: Finding 10-K filings for narrative text...")
    filing_urls = extractor.get_recent_filings(companies, years)
    
    # Step 3: Extract narrative text from 10-K filings
    print("\nStep 3: Extracting narrative text from 10-K filings...")
    narrative_data = {}

    for ticker, cik in companies.items():
        narrative_data[ticker] = {}
        for year in years:
            # Always use direct URL for each (ticker, year)
            sections = extractor.extract_narrative_text(
                ticker, cik, accession="", year=year
            )
            if sections:
                narrative_data[ticker][year] = sections
    
    # Step 4: Save raw data
    print("\nStep 4: Saving raw data...")
    extractor.save_raw_data(company_data)
    extractor.save_narrative_data(narrative_data)
    
    # Step 5: Create enhanced chunks (structured + narrative)
    print("\nStep 5: Creating enhanced text chunks for RAG...")
    chunks = extractor.create_text_chunks_for_rag(company_data, narrative_data, years)
    
    # Step 6: Save chunks
    print("\nStep 6: Saving chunks for RAG pipeline...")
    chunks_path = extractor.save_chunks_for_rag(chunks)
    
    print(f"\n✅ Enhanced data acquisition complete!")
    print(f"📊 Total chunks created: {len(chunks)}")
    print(f"📁 Chunks saved to: {chunks_path}")
    
    # Show sample chunks
    print("\n📋 Sample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"{i+1}. [{chunk['company']} {chunk['year']} - {chunk['type']}] {chunk['content'][:100]}...")
    
    return chunks

if __name__ == "__main__":
    main()


