import json
import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from collections import defaultdict
import re

@dataclass
class RetrievalResult:
    content: str
    company: str
    year: str
    metric: str
    value: float
    similarity_score: float
    chunk_id: int

class SECFinancialRAG:
    def __init__(self):
        print("🚀 Loading SEC Financial RAG Pipeline...")
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks = []
        self.raw_chunks = []  # Keep original data for debugging
        print("✅ Ready!")

    def deduplicate_chunks(self, raw_chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate metrics and keep the highest value (usually annual total)"""
        print("🔄 Deduplicating chunks...")
        
        # Group chunks by (company, year, metric)
        grouped = defaultdict(list)
        for chunk in raw_chunks:
            key = (chunk['company'], chunk['year'], chunk['metric'])
            grouped[key].append(chunk)
        
        # For each group, pick the best chunk
        deduplicated = []
        for key, chunk_group in grouped.items():
            if len(chunk_group) == 1:
                deduplicated.append(chunk_group[0])
            else:
                # Pick chunk with highest value (usually the annual total)
                best_chunk = max(chunk_group, key=lambda x: x.get('value', 0))
                deduplicated.append(best_chunk)
        
        print(f"📊 Deduplicated from {len(raw_chunks)} to {len(deduplicated)} chunks")
        return deduplicated

    def filter_relevant_metrics(self, chunks: List[Dict]) -> List[Dict]:
        """ENHANCED: Filter for both financial metrics AND narrative content"""
        print("🔄 Filtering for relevant financial metrics and narrative content...")
        
        # Keep ALL narrative text chunks (don't filter these)
        narrative_chunks = [c for c in chunks if c.get('type') == 'narrative_text']
        print(f"📝 Keeping all {len(narrative_chunks)} narrative chunks")
        
        # Filter financial metrics (as before)
        financial_chunks = []
        for chunk in chunks:
            if chunk.get('type') == 'financial_metric':
                # Key financial terms
                important_terms = [
                    'revenue', 'income', 'profit', 'loss', 'expense', 
                    'research', 'development', 'operating', 'gross', 'net'
                ]
                
                # Get metric and label safely
                metric = str(chunk.get('metric', '')).lower()
                label = str(chunk.get('label', '')).lower()
                content = str(chunk.get('content', '')).lower()
                
                # Keep if any important term appears
                if any(term in metric + ' ' + label + ' ' + content for term in important_terms):
                    financial_chunks.append(chunk)
        
        print(f"💰 Filtered to {len(financial_chunks)} relevant financial chunks")
        
        # Combine both types
        filtered_chunks = narrative_chunks + financial_chunks
        print(f"📊 Total filtered chunks: {len(filtered_chunks)} ({len(narrative_chunks)} narrative + {len(financial_chunks)} financial)")
        
        return filtered_chunks



    def load_clean_chunks(self, chunks_path: str = "data/rag_chunks.json"):
        """Load and process clean chunks from SEC API data"""
        print(f"📂 Loading chunks from {chunks_path}...")
        
        try:
            with open(chunks_path, 'r') as f:
                raw_chunks = json.load(f)
            
            print(f"📊 Loaded {len(raw_chunks)} raw chunks")
            self.raw_chunks = raw_chunks
            
            # Step 1: Filter for relevant metrics only
            relevant_chunks = self.filter_relevant_metrics(raw_chunks)
            
            # Step 2: Deduplicate similar metrics
            processed_chunks_data = self.deduplicate_chunks(relevant_chunks)
            
            # Step 3: Convert to RetrievalResult objects
            processed_chunks = []
            for i, chunk_data in enumerate(processed_chunks_data):
                chunk = RetrievalResult(
                    content=chunk_data['content'],
                    company=chunk_data['company'],
                    year=chunk_data['year'],
                    metric=chunk_data.get('metric', 'unknown'),
                    value=chunk_data.get('value', 0),
                    similarity_score=0.0,
                    chunk_id=i
                )
                processed_chunks.append(chunk)
            
            self.chunks = processed_chunks
            print(f"✅ Processed {len(self.chunks)} clean, deduplicated chunks for RAG")
            return True
            
        except FileNotFoundError:
            print(f"❌ Chunks file not found at {chunks_path}")
            print("   Please run data_acquisition.py first!")
            return False
        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            return False

    def build_index(self):
        """Build vector index from loaded chunks"""
        if not self.chunks:
            print("❌ No chunks loaded! Please run load_clean_chunks() first.")
            return
        
        print(f"🔄 Creating embeddings for {len(self.chunks)} chunks...")
        
        # Extract text content
        texts = [chunk.content for chunk in self.chunks]
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        
        print("✅ Vector index built successfully!")

    def smart_retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Smart retrieval with company detection and filtering"""
        if not self.chunks:
            print("❌ No chunks available. Please load data first!")
            return []
        
        # Extract company from query
        query_lower = query.lower()
        company_hints = {
            'microsoft': 'MSFT', 'msft': 'MSFT',
            'google': 'GOOGL', 'googl': 'GOOGL', 'alphabet': 'GOOGL',
            'nvidia': 'NVDA', 'nvda': 'NVDA'
        }
        
        detected_company = None
        for hint, company in company_hints.items():
            if hint in query_lower:
                detected_company = company
                break
        
        # Extract year from query
        year_match = re.search(r'(202[0-9])', query)
        detected_year = year_match.group(1) if year_match else None
        
        print(f"🔍 Detected: Company='{detected_company}', Year='{detected_year}'")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search with more candidates for filtering
        search_k = min(top_k * 10, len(self.chunks))
        scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)
        
        results = []
        company_matched = []
        other_results = []
        
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks[idx]
            chunk.similarity_score = float(score)
            
            # Prioritize chunks from detected company
            if detected_company and chunk.company == detected_company:
                # Apply year filter if specified
                if detected_year is None or str(chunk.year) == detected_year:
                    company_matched.append(chunk)
            else:
                other_results.append(chunk)
        
        # Combine results: prioritize company matches, then others
        if detected_company:
            results = company_matched[:top_k]
            # Fill remaining slots with other results if needed
            remaining_slots = top_k - len(results)
            if remaining_slots > 0:
                results.extend(other_results[:remaining_slots])
        else:
            results = (company_matched + other_results)[:top_k]
        
        return results

    def retrieve(self, query: str, top_k: int = 5, 
                 company_filter: Optional[str] = None, 
                 year_filter: Optional[str] = None) -> List[RetrievalResult]:
        """Standard retrieve method with explicit filters"""
        if not self.chunks:
            print("❌ No chunks available. Please load data first!")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search with extra candidates for filtering
        search_k = min(top_k * 5, len(self.chunks))
        scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks[idx]
            
            # Apply company filter
            if company_filter and chunk.company.upper() != company_filter.upper():
                continue
            
            # Apply year filter
            if year_filter and str(chunk.year) != str(year_filter):
                continue
            
            # Update similarity score
            chunk.similarity_score = float(score)
            results.append(chunk)
            
            if len(results) >= top_k:
                break
        
        return results

    def get_stats(self) -> Dict:
        """Get statistics about loaded chunks"""
        if not self.chunks:
            return {}
        
        companies = set(chunk.company for chunk in self.chunks)
        years = set(chunk.year for chunk in self.chunks)
        metrics = set(chunk.metric for chunk in self.chunks)
        
        return {
            'total_chunks': len(self.chunks),
            'companies': sorted(companies),
            'years': sorted(years),
            'unique_metrics': len(metrics),
            'sample_metrics': sorted(list(metrics))[:10]
        }

    def initialize(self, chunks_path: str = "data/rag_chunks.json"):
        """Complete initialization: load chunks and build index"""
        success = self.load_clean_chunks(chunks_path)
        if success:
            self.build_index()
            # Show stats
            stats = self.get_stats()
            print(f"\n📈 Dataset Stats:")
            print(f"   Companies: {stats['companies']}")
            print(f"   Years: {stats['years']}")
            print(f"   Total chunks: {stats['total_chunks']}")
            print(f"   Unique metrics: {stats['unique_metrics']}")
            return True
        return False

# Test the improved pipeline
if __name__ == "__main__":
    print("🧪 Testing Improved SEC Financial RAG Pipeline\n")
    
    # Initialize pipeline
    rag = SECFinancialRAG()
    
    # Load data and build index
    if not rag.initialize():
        print("❌ Failed to initialize RAG pipeline")
        exit(1)
    
    # Test queries with smart retrieval
    test_queries = [
        "Microsoft total revenue 2023",
        "NVIDIA revenue 2024", 
        "Google operating income 2023",
        "MSFT 2022 research and development expense"
    ]
    
    print("\n" + "="*60)
    print("🔍 TESTING SMART RETRIEVAL (Company Detection)")
    print("="*60)
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-" * 50)
        
        results = rag.smart_retrieve(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result.company} {result.year}] Score: {result.similarity_score:.3f}")
            print(f"   Value: ${result.value:,.0f}")
            print(f"   {result.content}")
    
    print("\n" + "="*60)
    print("🔍 TESTING EXPLICIT FILTERING")
    print("="*60)
    
    # Test explicit filtering
    test_cases = [
        ("total revenue", "MSFT", "2023"),
        ("operating income", "GOOGL", "2023"),
        ("revenue", "NVDA", "2024")
    ]
    
    for metric, company, year in test_cases:
        print(f"\n❓ Query: '{metric}' | Company: {company} | Year: {year}")
        print("-" * 50)
        
        results = rag.retrieve(metric, top_k=2, company_filter=company, year_filter=year)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result.company} {result.year}] Score: {result.similarity_score:.3f}")
            print(f"   Value: ${result.value:,.0f}")
            print(f"   {result.content}")
    
    print("\n✅ Testing complete!")


# import json
# import numpy as np
# from typing import List, Dict, Optional
# from dataclasses import dataclass
# from sentence_transformers import SentenceTransformer
# import faiss
# from tqdm import tqdm

# @dataclass
# class RetrievalResult:
#     content: str
#     company: str
#     year: str
#     similarity_score: float
#     chunk_id: int

# class SECFinancialRAG:
#     def __init__(self):
#         print("🚀 Loading SEC Financial RAG Pipeline...")
#         self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')
#         self.embedding_dim = self.model.get_sentence_embedding_dimension()
#         self.index = faiss.IndexFlatIP(self.embedding_dim)
#         self.chunks = []
#         print("✅ Ready!")

#     def load_clean_chunks(self, chunks_path: str = "data/rag_chunks.json"):
#         """Load the clean chunks from SEC API data"""
#         print(f"📂 Loading chunks from {chunks_path}...")
        
#         try:
#             with open(chunks_path, 'r') as f:
#                 raw_chunks = json.load(f)
            
#             print(f"📊 Loaded {len(raw_chunks)} raw chunks")
            
#             # Convert to RetrievalResult objects
#             processed_chunks = []
#             for i, chunk_data in enumerate(raw_chunks):
#                 chunk = RetrievalResult(
#                     content=chunk_data['content'],
#                     company=chunk_data['company'],
#                     year=chunk_data['year'],
#                     similarity_score=0.0,
#                     chunk_id=i
#                 )
#                 processed_chunks.append(chunk)
            
#             self.chunks = processed_chunks
#             print(f"✅ Processed {len(self.chunks)} chunks for RAG")
#             return True
            
#         except FileNotFoundError:
#             print(f"❌ Chunks file not found at {chunks_path}")
#             print("   Please run data_acquisition.py first!")
#             return False
#         except Exception as e:
#             print(f"❌ Error loading chunks: {e}")
#             return False

#     def build_index(self):
#         """Build vector index from loaded chunks"""
#         if not self.chunks:
#             print("❌ No chunks loaded! Please run load_clean_chunks() first.")
#             return
        
#         print(f"🔄 Creating embeddings for {len(self.chunks)} chunks...")
        
#         # Extract text content
#         texts = [chunk.content for chunk in self.chunks]
        
#         # Generate embeddings in batches
#         batch_size = 32
#         embeddings = self.model.encode(
#             texts, 
#             batch_size=batch_size,
#             show_progress_bar=True, 
#             normalize_embeddings=True
#         )
        
#         # Build FAISS index
#         self.index = faiss.IndexFlatIP(self.embedding_dim)
#         self.index.add(embeddings.astype(np.float32))
        
#         print("✅ Vector index built successfully!")

#     def retrieve(self, query: str, top_k: int = 5, 
#                  company_filter: Optional[str] = None, 
#                  year_filter: Optional[str] = None) -> List[RetrievalResult]:
#         """Retrieve relevant chunks with optional filtering"""
#         if not self.chunks:
#             print("❌ No chunks available. Please load data first!")
#             return []
        
#         # Generate query embedding
#         query_embedding = self.model.encode([query], normalize_embeddings=True)
        
#         # Search with extra candidates for filtering
#         search_k = min(top_k * 3, len(self.chunks))
#         scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)
        
#         results = []
#         for score, idx in zip(scores[0], indices[0]):
#             chunk = self.chunks[idx]
            
#             # Apply company filter
#             if company_filter and chunk.company.upper() != company_filter.upper():
#                 continue
            
#             # Apply year filter
#             if year_filter and str(chunk.year) != str(year_filter):
#                 continue
            
#             # Update similarity score
#             chunk.similarity_score = float(score)
#             results.append(chunk)
            
#             if len(results) >= top_k:
#                 break
        
#         return results

#     def initialize(self, chunks_path: str = "data/rag_chunks.json"):
#         """Complete initialization: load chunks and build index"""
#         success = self.load_clean_chunks(chunks_path)
#         if success:
#             self.build_index()
#             return True
#         return False

# # Test the new pipeline
# if __name__ == "__main__":
#     print("🧪 Testing SEC Financial RAG Pipeline\n")
    
#     # Initialize pipeline
#     rag = SECFinancialRAG()
    
#     # Load data and build index
#     if not rag.initialize():
#         print("❌ Failed to initialize RAG pipeline")
#         exit(1)
    
#     # Test queries
#     test_queries = [
#         "Microsoft total revenue 2023",
#         "NVIDIA revenue 2024", 
#         "Google operating income 2023"
#     ]
    
#     print("\n" + "="*50)
#     print("🔍 TESTING CLEAN DATA RETRIEVAL")
#     print("="*50)
    
#     for query in test_queries:
#         print(f"\n❓ Query: {query}")
#         print("-" * 40)
        
#         results = rag.retrieve(query, top_k=3)
        
#         for i, result in enumerate(results, 1):
#             print(f"{i}. [{result.company} {result.year}] Score: {result.similarity_score:.3f}")
#             print(f"   {result.content}")
    
#     print("\n✅ Testing complete!")
