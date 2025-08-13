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





















# the one with issue 2 resolved - Keyword Search + Filters


# """
# Enhanced SEC Data Acquisition with Semantic Chunking 
# """

# import requests
# import pandas as pd
# import json
# import os
# import re
# from typing import Dict, List, Tuple, Optional
# from bs4 import BeautifulSoup
# import time
# from dataclasses import dataclass

# class EnhancedSECExtractor:
#     def __init__(self):
#         self.headers = {
#             'User-Agent': 'Shreya jiminiepabo3130@gmail.com'  
#         }
        
#         # Enhanced financial patterns for extraction
#         self.financial_patterns = {
#             'revenue': [
#                 r'revenue.*?(\$[\d,\.]+\s*(?:billion|million|thousand))',
#                 r'(?:net\s+)?revenue.*?(\$[\d,\.]+\s*(?:billion|million))',
#                 r'total\s+revenue.*?(\$[\d,\.]+\s*(?:billion|million))',
#                 r'(?:cloud|segment)\s+revenue.*?(\$[\d,\.]+\s*(?:billion|million))'
#             ],
#             'margin': [
#                 r'(?:operating|gross|net)\s+margin.*?(\d+\.?\d*%)',
#                 r'margin.*?(\d+\.?\d*%)',
#                 r'margin\s+(?:was|of).*?(\d+\.?\d*%)'
#             ],
#             'growth': [
#                 r'(?:increased|grew|growth).*?(\d+\.?\d*%)',
#                 r'(?:decreased|declined|decline).*?(\d+\.?\d*%)',
#                 r'(?:up|down).*?(\d+\.?\d*%)',
#                 r'(?:year-over-year|YoY).*?(\d+\.?\d*%)'
#             ],
#             'profit': [
#                 r'(?:net\s+)?income.*?(\$[\d,\.]+\s*(?:billion|million))',
#                 r'profit.*?(\$[\d,\.]+\s*(?:billion|million))',
#                 r'earnings.*?(\$[\d,\.]+\s*(?:billion|million))',
#                 r'operating\s+income.*?(\$[\d,\.]+\s*(?:billion|million))'
#             ]
#         }
        
#         # Entity patterns for product/segment identification
#         self.entity_patterns = [
#             r'(?i)\b(?:cloud|azure|aws|gcp|google cloud|data center|datacenter|AI|artificial intelligence)\b',
#             r'(?i)\b(?:gaming|xbox|surface|office|windows|teams|outlook)\b',
#             r'(?i)\b(?:youtube|search|advertising|ads|adwords|adsense)\b',
#             r'(?i)\b(?:gpu|graphics|automotive|professional visualization|geforce)\b',
#             r'(?i)\b(?:segment|division|business unit|product line|operating segment)\b'
#         ]

#     def get_correct_filing_urls(self) -> Dict[Tuple[str, str], str]:
#         """Returns working URLs for 10-K filings"""
#         return {
#             # Microsoft URLs
#             ('MSFT', '2022'): 'https://www.sec.gov/Archives/edgar/data/789019/000156459022026876/msft-10k_20220630.htm',
#             ('MSFT', '2023'): 'https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm',
#             ('MSFT', '2024'): 'https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm',
            
#             # Google/Alphabet URLs
#             ('GOOGL', '2022'): 'https://www.sec.gov/Archives/edgar/data/1652044/000165204423000016/goog-20221231.htm',
#             ('GOOGL', '2023'): 'https://www.sec.gov/Archives/edgar/data/1652044/000165204424000022/goog-20231231.htm',
#             ('GOOGL', '2024'): 'https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm',
            
#             # NVIDIA URLs
#             ('NVDA', '2022'): 'https://www.sec.gov/Archives/edgar/data/1045810/000104581022000036/nvda-20220130.htm',
#             ('NVDA', '2023'): 'https://www.sec.gov/Archives/edgar/data/1045810/000104581023000017/nvda-20230129.htm',
#             ('NVDA', '2024'): 'https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm'
#         }

#     def clean_html_content(self, html_content: str) -> str:
#         """Enhanced HTML cleaning for SEC filings"""
#         try:
#             soup = BeautifulSoup(html_content, 'html.parser')
            
#             # Remove XBRL and hidden content
#             for tag in soup.find_all(['ix:hidden', 'ix:nonfraction', 'ix:nonnumeric', 'script', 'style']):
#                 tag.decompose()
            
#             # Get text content
#             text = soup.get_text()
            
#             # Clean whitespace and normalize
#             lines = (line.strip() for line in text.splitlines())
#             text = ' '.join(line for line in lines if line)
            
#             return text
#         except Exception as e:
#             print(f"❌ Error cleaning HTML: {e}")
#             return html_content

#     def extract_financial_mentions(self, content: str) -> List[Dict]:
#         """Extract financial mentions with context"""
#         mentions = []
        
#         for metric_type, patterns in self.financial_patterns.items():
#             for pattern in patterns:
#                 try:
#                     matches = re.finditer(pattern, content, re.IGNORECASE)
#                     for match in matches:
#                         # Get context around the match
#                         start_pos = max(0, match.start() - 150)
#                         end_pos = min(len(content), match.end() + 150)
#                         context = content[start_pos:end_pos].strip()
                        
#                         mentions.append({
#                             'metric_type': metric_type,
#                             'value': match.group(1) if match.groups() else match.group(0),
#                             'context': context,
#                             'position': match.start()
#                         })
#                 except Exception:
#                     continue
        
#         return mentions

#     def extract_temporal_markers(self, content: str) -> List[str]:
#         """Extract temporal references"""
#         temporal_patterns = [
#             r'\b(20\d{2})\b',
#             r'\b(Q[1-4])\b',
#             r'\b(first|second|third|fourth)\s+quarter\b',
#             r'\b(fiscal\s+year|FY)\s*(\d{4}|\d{2})\b',
#             r'\b(year\s+ended?|year-over-year|YoY)\b'
#         ]
        
#         markers = []
#         for pattern in temporal_patterns:
#             try:
#                 matches = re.findall(pattern, content, re.IGNORECASE)
#                 for match in matches:
#                     if isinstance(match, tuple):
#                         markers.extend([m for m in match if m])
#                     else:
#                         markers.append(match)
#             except:
#                 continue
        
#         return list(set(markers))

#     def extract_entity_references(self, content: str) -> List[str]:
#         """Extract entity references"""
#         entities = []
#         for pattern in self.entity_patterns:
#             try:
#                 matches = re.findall(pattern, content)
#                 entities.extend([match.lower() for match in matches])
#             except:
#                 continue
        
#         return list(set(entities))

#     def create_enriched_chunk(self, content: str, company: str, year: str, chunk_id: str) -> Dict:
#         """Create an enriched chunk with all the metadata for good retrieval"""
        
#         # Extract all the enriching data
#         financial_mentions = self.extract_financial_mentions(content)
#         temporal_markers = self.extract_temporal_markers(content)
#         entity_references = self.extract_entity_references(content)
        
#         # Create the enriched chunk
#         return {
#             'chunk_id': chunk_id,
#             'content': content,
#             'company': company,
#             'year': year,
#             'word_count': len(content.split()),
#             'char_count': len(content),
            
#             # Rich metadata for retrieval
#             'financial_mentions': financial_mentions,
#             'temporal_markers': temporal_markers,
#             'entity_references': entity_references,
            
#             # Searchable fields
#             'has_financial_data': len(financial_mentions) > 0,
#             'has_temporal_refs': len(temporal_markers) > 0,
#             'mention_count': len(financial_mentions),
            
#             # For filtering and search
#             'searchable_text': f"{company} {year} {content} {' '.join(entity_references)}".lower(),
            
#             # Metadata
#             'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
#         }

#     def create_semantic_chunks(self, text: str, company: str, year: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
#         """Create semantically meaningful chunks without hard-coded sections"""
        
#         # Clean and normalize the text
#         clean_text = re.sub(r'\s+', ' ', text).strip()
        
#         # Split into sentences for better semantic boundaries
#         sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        
#         chunks = []
#         current_chunk = []
#         current_word_count = 0
        
#         for sentence in sentences:
#             sentence_words = len(sentence.split())
            
#             # If adding this sentence would exceed chunk size, save current chunk
#             if current_word_count + sentence_words > chunk_size and current_chunk:
#                 chunk_content = ' '.join(current_chunk)
                
#                 if len(chunk_content.strip()) > 200:  # Only keep substantial chunks
#                     chunk_data = self.create_enriched_chunk(
#                         content=chunk_content,
#                         company=company,
#                         year=year,
#                         chunk_id=f"{company}_{year}_chunk_{len(chunks)}"
#                     )
#                     chunks.append(chunk_data)
                
#                 # Start new chunk with overlap
#                 if overlap > 0 and len(current_chunk) > overlap:
#                     overlap_sentences = current_chunk[-overlap//10:]  # Approximate sentence overlap
#                     current_chunk = overlap_sentences
#                     current_word_count = sum(len(s.split()) for s in current_chunk)
#                 else:
#                     current_chunk = []
#                     current_word_count = 0
            
#             current_chunk.append(sentence)
#             current_word_count += sentence_words
        
#         # Don't forget the last chunk
#         if current_chunk:
#             chunk_content = ' '.join(current_chunk)
#             if len(chunk_content.strip()) > 200:
#                 chunk_data = self.create_enriched_chunk(
#                     content=chunk_content,
#                     company=company,
#                     year=year,
#                     chunk_id=f"{company}_{year}_chunk_{len(chunks)}"
#                 )
#                 chunks.append(chunk_data)
        
#         return chunks

#     def create_company_year_documents_semantic(self, companies: Dict[str, str], years: List[str]) -> Dict[str, Dict]:
#         """Create company-year documents with semantic chunking (no sections)"""
#         print("🔄 Creating Company-Year Documents with Semantic Chunking...")
        
#         filing_urls = self.get_correct_filing_urls()
#         company_year_docs = {}
        
#         for ticker, cik in companies.items():
#             company_year_docs[ticker] = {}
            
#             for year in years:
#                 print(f"\n📊 Processing {ticker} {year}...")
#                 url_key = (ticker, year)
                
#                 if url_key not in filing_urls:
#                     print(f"⚠️ No URL available for {ticker} {year}")
#                     continue
                    
#                 url = filing_urls[url_key]
                
#                 try:
#                     response = requests.get(url, headers=self.headers, timeout=60)
#                     if response.status_code != 200:
#                         print(f"❌ Failed to fetch {ticker} {year}: {response.status_code}")
#                         continue
                    
#                     # Clean the content
#                     cleaned_text = self.clean_html_content(response.text)
                    
#                     # Skip if content is too small
#                     if len(cleaned_text) < 10000:
#                         print(f"⚠️ Content too small for {ticker} {year}")
#                         continue
                    
#                     # Take the most relevant part (middle portion typically has MD&A and business info)
#                     start_idx = len(cleaned_text) // 6  # Skip early boilerplate
#                     end_idx = 5 * len(cleaned_text) // 6  # Skip late boilerplate
#                     relevant_text = cleaned_text[start_idx:end_idx]
                    
#                     # Create semantic chunks
#                     chunks = self.create_semantic_chunks(
#                         text=relevant_text,
#                         company=ticker,
#                         year=year,
#                         chunk_size=450,  # Good size for retrieval
#                         overlap=80       # Good overlap for context preservation
#                     )
                    
#                     # Create unified document
#                     company_year_doc = {
#                         'company': ticker,
#                         'year': year,
#                         'url': url,
#                         'document_id': f"{ticker}_{year}",
#                         'chunks': chunks,
#                         'total_chunks': len(chunks),
#                         'total_length': len(relevant_text),
#                         'processing_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                        
#                         # Aggregate statistics
#                         'total_financial_mentions': sum(len(chunk['financial_mentions']) for chunk in chunks),
#                         'chunks_with_financial_data': sum(1 for chunk in chunks if chunk['has_financial_data']),
#                         'unique_entities': list(set([entity for chunk in chunks for entity in chunk['entity_references']])),
#                         'temporal_coverage': list(set([marker for chunk in chunks for marker in chunk['temporal_markers']])),
                        
#                         # For easy access
#                         'sample_entities': list(set([entity for chunk in chunks for entity in chunk['entity_references']]))[:10],
#                         'sample_financial_mentions': [mention['metric_type'] for chunk in chunks for mention in chunk['financial_mentions']][:10]
#                     }
                    
#                     company_year_docs[ticker][year] = company_year_doc
                    
#                     print(f"✅ {ticker} {year}: Created {len(chunks)} semantic chunks")
#                     print(f"   📊 Financial mentions: {company_year_doc['total_financial_mentions']}")
#                     print(f"   🏷️  Entities found: {len(company_year_doc['unique_entities'])}")
#                     print(f"   📄 Content length: {len(relevant_text):,} chars")
                    
#                     time.sleep(1)
                    
#                 except Exception as e:
#                     print(f"❌ Error processing {ticker} {year}: {e}")
#                     continue
        
#         return company_year_docs

#     def create_retrieval_chunks(self, company_year_docs: Dict) -> List[Dict]:
#         """Flatten all chunks for retrieval system"""
#         all_chunks = []
        
#         for ticker, years_data in company_year_docs.items():
#             for year, doc_data in years_data.items():
#                 for chunk in doc_data['chunks']:
#                     # Add document-level metadata to each chunk
#                     chunk['document_context'] = {
#                         'total_chunks_in_doc': doc_data['total_chunks'],
#                         'doc_financial_mentions': doc_data['total_financial_mentions'],
#                         'doc_entities': doc_data['unique_entities'][:10],  # Limit size
#                         'doc_temporal_coverage': doc_data['temporal_coverage']
#                     }
#                     all_chunks.append(chunk)
        
#         return all_chunks

#     def save_company_year_documents(self, company_year_docs: Dict, save_dir: str = "data/company_year_docs") -> Tuple[List[str], str]:
#         """Save each company-year as separate JSON file"""
#         os.makedirs(save_dir, exist_ok=True)
        
#         saved_files = []
        
#         for ticker, years_data in company_year_docs.items():
#             for year, doc_data in years_data.items():
#                 filename = f"{ticker}_{year}_semantic.json"
#                 filepath = os.path.join(save_dir, filename)
                
#                 with open(filepath, 'w') as f:
#                     json.dump(doc_data, f, indent=2)
                
#                 saved_files.append(filepath)
#                 print(f"💾 Saved {ticker} {year} semantic document to {filename}")
        
#         # Create a master index
#         index_file = os.path.join(save_dir, "semantic_index.json")
#         index_data = {
#             'total_documents': len(saved_files),
#             'companies': list(company_year_docs.keys()),
#             'years': list(set([year for years_data in company_year_docs.values() for year in years_data.keys()])),
#             'files': [os.path.basename(f) for f in saved_files],
#             'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
#             'document_structure': {
#                 'chunking_method': 'semantic_sentence_based',
#                 'sections_included': False,
#                 'chunk_size': 450,
#                 'chunk_overlap': 80,
#                 'features': ['financial_mentions', 'temporal_markers', 'entity_references']
#             }
#         }
        
#         with open(index_file, 'w') as f:
#             json.dump(index_data, f, indent=2)
        
#         print(f"📋 Created semantic index at {index_file}")
#         return saved_files, index_file

#     def load_company_year_document(self, company: str, year: str, load_dir: str = "data/company_year_docs") -> Optional[Dict]:
#         """Load a specific company-year document"""
#         filename = f"{company}_{year}_semantic.json"
#         filepath = os.path.join(load_dir, filename)
        
#         try:
#             with open(filepath, 'r') as f:
#                 return json.load(f)
#         except FileNotFoundError:
#             print(f"⚠️ Document not found: {filepath}")
#             return None
#         except Exception as e:
#             print(f"❌ Error loading document: {e}")
#             return None

# def main():
#     print("🚀 Enhanced SEC Data Acquisition with Semantic Chunking (No Sections)\n")
    
#     companies = {
#         'MSFT': '789019',
#         'GOOGL': '1652044', 
#         'NVDA': '1045810'
#     }
    
#     years = ['2022', '2023', '2024']
    
#     extractor = EnhancedSECExtractor()
    
#     # Create company-year documents with semantic chunking
#     print("Step 1: Creating company-year documents with semantic chunking...")
#     company_year_docs = extractor.create_company_year_documents_semantic(companies, years)
    
#     # Save documents
#     print("\nStep 2: Saving company-year documents...")
#     saved_files, index_file = extractor.save_company_year_documents(company_year_docs)
    
#     # Create retrieval-ready chunks
#     print("\nStep 3: Creating retrieval-ready chunks...")
#     retrieval_chunks = extractor.create_retrieval_chunks(company_year_docs)
    
#     # Save retrieval chunks
#     chunks_path = "data/semantic_retrieval_chunks.json"
#     os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
#     with open(chunks_path, 'w') as f:
#         json.dump(retrieval_chunks, f, indent=2)
    
#     print(f"\n✅ Semantic Company-Year Data Acquisition Complete!")
#     print(f"📊 Total company-year documents: {sum(len(years_data) for years_data in company_year_docs.values())}")
#     print(f"📁 Documents saved to: data/company_year_docs/")
#     print(f"📋 Semantic index: {index_file}")
#     print(f"🔍 Retrieval chunks: {len(retrieval_chunks)} saved to {chunks_path}")
    
#     # Show sample document structure
#     if company_year_docs:
#         sample_company = list(company_year_docs.keys())[0]
#         sample_year = list(company_year_docs[sample_company].keys())[0]
#         sample_doc = company_year_docs[sample_company][sample_year]
        
#         print(f"\n📋 Sample Document Structure ({sample_company} {sample_year}):")
#         print(f"  - Method: Semantic chunking ")
#         print(f"  - Total chunks: {len(sample_doc['chunks'])}")
#         print(f"  - Financial mentions: {sample_doc['total_financial_mentions']}")
#         print(f"  - Entities found: {sample_doc['sample_entities'][:5]}")
#         print(f"  - Content length: {sample_doc['total_length']:,} chars")
#         print(f"  - Chunks with financial data: {sample_doc['chunks_with_financial_data']}")
    
#     return company_year_docs, retrieval_chunks

# if __name__ == "__main__":
#     main()
