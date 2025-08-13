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










# the one with issue3,4,5 resolved - Unified Parsing Strategy,Improved Cleaning and Parsing,Single Source for Both Analysis Types 

# import json
# import numpy as np
# from typing import List, Dict, Optional, Set, Union
# from dataclasses import dataclass
# from sentence_transformers import SentenceTransformer
# import faiss
# from rank_bm25 import BM25Okapi
# import re
# import os
# from collections import defaultdict


# @dataclass
# class EnhancedRetrievalResult:
#     content: str
#     company: str
#     year: str
#     chunk_id: str
#     similarity_score: float = 0.0
#     bm25_score: float = 0.0
#     combined_score: float = 0.0
    
#     # Rich metadata from new data structure
#     financial_mentions: List[Dict] = None
#     entity_references: List[str] = None
#     temporal_markers: List[str] = None
#     has_financial_data: bool = False
#     mention_count: int = 0
    
#     # Retrieval method used
#     retrieval_method: str = "semantic"


# class HybridFinancialRAG:
#     def __init__(self):
#         print("🚀 Loading Hybrid SEC Financial RAG Pipeline...")
        
#         # Semantic search components
#         self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')
#         self.embedding_dim = self.model.get_sentence_embedding_dimension()
#         self.semantic_index = faiss.IndexFlatIP(self.embedding_dim)
        
#         # Keyword search components
#         self.bm25_index = None
#         self.tokenized_docs = []
        
#         # Data storage
#         self.chunks = []
#         self.chunk_lookup = {}  # For fast ID-based retrieval
        
#         print("✅ Ready!")


#     def load_semantic_chunks(self, chunks_path: str = "data/semantic_retrieval_chunks.json"):
#         """Load semantic chunks from enhanced data acquisition"""
#         print(f"📂 Loading semantic chunks from {chunks_path}...")
        
#         try:
#             with open(chunks_path, 'r') as f:
#                 raw_chunks = json.load(f)
            
#             print(f"📊 Loaded {len(raw_chunks)} semantic chunks")
            
#             # Convert to enhanced retrieval results
#             processed_chunks = []
#             for chunk_data in raw_chunks:
#                 chunk = EnhancedRetrievalResult(
#                     content=chunk_data['content'],
#                     company=chunk_data['company'],
#                     year=chunk_data['year'],
#                     chunk_id=chunk_data['chunk_id'],
                    
#                     # Rich metadata
#                     financial_mentions=chunk_data.get('financial_mentions', []),
#                     entity_references=chunk_data.get('entity_references', []),
#                     temporal_markers=chunk_data.get('temporal_markers', []),
#                     has_financial_data=chunk_data.get('has_financial_data', False),
#                     mention_count=chunk_data.get('mention_count', 0)
#                 )
#                 processed_chunks.append(chunk)
#                 self.chunk_lookup[chunk.chunk_id] = chunk
            
#             self.chunks = processed_chunks
#             print(f"✅ Processed {len(self.chunks)} enhanced chunks")
            
#             # Show data statistics
#             companies = set(chunk.company for chunk in self.chunks)
#             years = set(chunk.year for chunk in self.chunks)
#             chunks_with_financial = sum(1 for chunk in self.chunks if chunk.has_financial_data)
            
#             print(f"📈 Dataset Overview:")
#             print(f"   Companies: {sorted(companies)}")
#             print(f"   Years: {sorted(years)}")
#             print(f"   Chunks with financial data: {chunks_with_financial}/{len(self.chunks)}")
            
#             return True
            
#         except FileNotFoundError:
#             print(f"❌ Chunks file not found at {chunks_path}")
#             print("   Please run the enhanced data_acquisition.py first!")
#             return False
#         except Exception as e:
#             print(f"❌ Error loading chunks: {e}")
#             return False


#     def build_semantic_index(self):
#         """Build FAISS semantic search index"""
#         if not self.chunks:
#             print("❌ No chunks loaded!")
#             return False
        
#         print(f"🔄 Building semantic index for {len(self.chunks)} chunks...")
        
#         # Create searchable text combining content + metadata
#         texts = []
#         for chunk in self.chunks:
#             # Combine content with entity references for richer semantic matching
#             searchable_text = chunk.content
#             if chunk.entity_references:
#                 searchable_text += " " + " ".join(chunk.entity_references)
#             texts.append(searchable_text)
        
#         # Generate embeddings
#         embeddings = self.model.encode(
#             texts,
#             batch_size=32,
#             show_progress_bar=True,
#             normalize_embeddings=True
#         )
        
#         # Build FAISS index
#         self.semantic_index = faiss.IndexFlatIP(self.embedding_dim)
#         self.semantic_index.add(embeddings.astype(np.float32))
        
#         print("✅ Semantic index built successfully!")
#         return True


#     def build_keyword_index(self):
#         """Build BM25 keyword search index"""
#         if not self.chunks:
#             print("❌ No chunks loaded!")
#             return False
        
#         print(f"🔄 Building BM25 keyword index for {len(self.chunks)} chunks...")
        
#         # Create searchable documents for BM25
#         # Include content + financial mentions + entities for comprehensive keyword matching
#         bm25_docs = []
#         for chunk in self.chunks:
#             doc_text = chunk.content
            
#             # Add financial mention contexts
#             if chunk.financial_mentions:
#                 financial_contexts = [mention.get('context', '') for mention in chunk.financial_mentions]
#                 doc_text += " " + " ".join(financial_contexts)
            
#             # Add entity references
#             if chunk.entity_references:
#                 doc_text += " " + " ".join(chunk.entity_references)
            
#             # Add temporal markers
#             if chunk.temporal_markers:
#                 doc_text += " " + " ".join(chunk.temporal_markers)
                
#             bm25_docs.append(doc_text)
        
#         # Tokenize documents for BM25
#         self.tokenized_docs = [doc.lower().split() for doc in bm25_docs]
        
#         # Build BM25 index with financial document optimized parameters
#         self.bm25_index = BM25Okapi(
#             self.tokenized_docs,
#             k1=1.2,  # Term frequency saturation parameter
#             b=0.75   # Document length normalization
#         )
        
#         print("✅ BM25 keyword index built successfully!")
#         return True


#     def semantic_search(self, query: str, top_k: int = 10) -> List[EnhancedRetrievalResult]:
#         """Pure semantic search"""
#         if not self.chunks or self.semantic_index.ntotal == 0:
#             return []
        
#         # Generate query embedding
#         query_embedding = self.model.encode([query], normalize_embeddings=True)
        
#         # Search
#         scores, indices = self.semantic_index.search(
#             query_embedding.astype(np.float32), 
#             min(top_k, len(self.chunks))
#         )
        
#         results = []
#         for score, idx in zip(scores[0], indices[0]):
#             chunk = self.chunks[idx]
#             result = EnhancedRetrievalResult(**chunk.__dict__)
#             result.similarity_score = float(score)
#             result.combined_score = float(score)  # FIX 1: Set combined = semantic for pure semantic
#             result.retrieval_method = "semantic"
#             results.append(result)
        
#         return results


#     def keyword_search(self, query: str, top_k: int = 10) -> List[EnhancedRetrievalResult]:
#         """Pure BM25 keyword search"""
#         if not self.chunks or not self.bm25_index:
#             return []
        
#         # Tokenize query
#         query_tokens = query.lower().split()
        
#         # Get BM25 scores
#         bm25_scores = self.bm25_index.get_scores(query_tokens)
        
#         # Get top-k results
#         top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
#         results = []
#         for idx in top_indices:
#             if bm25_scores[idx] > 0:  # Only include relevant results
#                 chunk = self.chunks[idx]
#                 result = EnhancedRetrievalResult(**chunk.__dict__)
#                 result.bm25_score = float(bm25_scores[idx])
#                 result.combined_score = float(bm25_scores[idx])  # FIX 1: Set combined = BM25 for pure keyword
#                 result.retrieval_method = "keyword"
#                 results.append(result)
        
#         return results


#     def hybrid_search(self, query: str, top_k: int = 10, 
#                      semantic_weight: float = 0.6, keyword_weight: float = 0.4) -> List[EnhancedRetrievalResult]:
#         """Hybrid search combining semantic and keyword approaches"""
        
#         # Get results from both methods
#         semantic_results = self.semantic_search(query, top_k * 2)
#         keyword_results = self.keyword_search(query, top_k * 2)
        
#         # Create lookup for combining scores
#         chunk_scores = {}
        
#         # Add semantic scores
#         for result in semantic_results:
#             chunk_scores[result.chunk_id] = {
#                 'chunk': result,
#                 'semantic_score': result.similarity_score,
#                 'keyword_score': 0.0
#             }
        
#         # Add keyword scores
#         for result in keyword_results:
#             if result.chunk_id in chunk_scores:
#                 chunk_scores[result.chunk_id]['keyword_score'] = result.bm25_score
#             else:
#                 chunk_scores[result.chunk_id] = {
#                     'chunk': result,
#                     'semantic_score': 0.0,
#                     'keyword_score': result.bm25_score
#                 }
        
#         # Normalize and combine scores
#         if chunk_scores:
#             # Normalize semantic scores (0-1 range)
#             semantic_scores = [data['semantic_score'] for data in chunk_scores.values()]
#             max_semantic = max(semantic_scores) if semantic_scores else 1
            
#             # Normalize keyword scores (0-1 range)  
#             keyword_scores = [data['keyword_score'] for data in chunk_scores.values()]
#             max_keyword = max(keyword_scores) if keyword_scores else 1
            
#             # Calculate combined scores
#             combined_results = []
#             for chunk_id, data in chunk_scores.items():
#                 result = EnhancedRetrievalResult(**data['chunk'].__dict__)
                
#                 # Normalize scores
#                 norm_semantic = data['semantic_score'] / max_semantic if max_semantic > 0 else 0
#                 norm_keyword = data['keyword_score'] / max_keyword if max_keyword > 0 else 0
                
#                 # Combine scores
#                 result.similarity_score = norm_semantic
#                 result.bm25_score = norm_keyword
#                 result.combined_score = (semantic_weight * norm_semantic) + (keyword_weight * norm_keyword)
#                 result.retrieval_method = "hybrid"
                
#                 combined_results.append(result)
            
#             # Sort by combined score
#             combined_results.sort(key=lambda x: x.combined_score, reverse=True)
#             return combined_results[:top_k]
        
#         return []


#     def apply_filters(self, results: List[EnhancedRetrievalResult], 
#                      company_filter: Optional[str] = None,
#                      year_filter: Optional[str] = None,
#                      has_financial_filter: Optional[bool] = None) -> List[EnhancedRetrievalResult]:
#         """Apply metadata filters to results"""
        
#         filtered = results
        
#         # Company filter
#         if company_filter:
#             company_upper = company_filter.upper()
#             filtered = [r for r in filtered if r.company.upper() == company_upper]
        
#         # Year filter
#         if year_filter:
#             year_str = str(year_filter)
#             filtered = [r for r in filtered if str(r.year) == year_str]
        
#         # Financial data filter
#         if has_financial_filter is not None:
#             filtered = [r for r in filtered if r.has_financial_data == has_financial_filter]
        
#         return filtered


#     def smart_retrieve(self, query: str, top_k: int = 5, method: str = "auto") -> List[EnhancedRetrievalResult]:
#         """Intelligent retrieval with automatic method selection"""
        
#         if not self.chunks:
#             print("❌ No chunks available!")
#             return []
        
#         query_lower = query.lower()
        
#         # Auto-detect optimal retrieval method
#         if method == "auto":
#             # FIX 2: More selective keyword triggers - only when company + year together
#             exact_patterns = [
#                 r'\b\d+\.?\d*%\b',  # Percentages
#                 r'\$[\d,\.]+',      # Dollar amounts  
#                 r'\b(MSFT|GOOGL|NVDA)\s+(202[0-4])\b',  # Company + Year together
#                 r'\b(clause|section)\s+\d+'  # Clauses/sections
#             ]
            
#             if any(re.search(pattern, query, re.IGNORECASE) for pattern in exact_patterns):
#                 method = "keyword"
#                 print(f"🔍 Auto-selected: keyword search (detected exact terms)")
#             else:
#                 method = "hybrid"
#                 print(f"🔍 Auto-selected: hybrid search")
        
#         # Auto-detect filters from query
#         company_filter = None
#         year_filter = None
        
#         # Detect company
#         company_hints = {
#             'microsoft': 'MSFT', 'msft': 'MSFT',
#             'google': 'GOOGL', 'googl': 'GOOGL', 'alphabet': 'GOOGL',
#             'nvidia': 'NVDA', 'nvda': 'NVDA'
#         }
        
#         for hint, company in company_hints.items():
#             if hint in query_lower:
#                 company_filter = company
#                 break
        
#         # Detect year
#         year_match = re.search(r'\b(202[0-4])\b', query)
#         if year_match:
#             year_filter = year_match.group(1)
        
#         print(f"🔍 Query analysis: company={company_filter}, year={year_filter}")
        
#         # Execute search based on method
#         if method == "semantic":
#             results = self.semantic_search(query, top_k * 3)
#         elif method == "keyword":
#             results = self.keyword_search(query, top_k * 3)
#         elif method == "hybrid":
#             results = self.hybrid_search(query, top_k * 3)
#         else:
#             raise ValueError(f"Unknown method: {method}")
        
#         # Apply detected filters
#         filtered_results = self.apply_filters(
#             results, 
#             company_filter=company_filter, 
#             year_filter=year_filter
#         )
        
#         return filtered_results[:top_k]


#     def retrieve_for_agent(self, query: str, top_k: int = 3) -> List[Dict]:
#         """Simplified retrieval method for agent use"""
#         results = self.smart_retrieve(query, top_k=top_k)
#         return [
#             {
#                 'content': r.content,
#                 'company': r.company,
#                 'year': r.year,
#                 'score': r.combined_score,
#                 'financial_mentions': r.financial_mentions,
#                 'source': f"{r.company}_{r.year}_chunk_{r.chunk_id.split('_')[-1]}"
#             }
#             for r in results
#         ]

#     def initialize(self, chunks_path: str = "data/semantic_retrieval_chunks.json"):
#         """Complete initialization: load data and build all indices"""
#         print("🚀 Initializing Hybrid Financial RAG...")
        
#         # Load data
#         if not self.load_semantic_chunks(chunks_path):
#             return False
        
#         # Build indices
#         semantic_success = self.build_semantic_index()
#         keyword_success = self.build_keyword_index()
        
#         if semantic_success and keyword_success:
#             print("\n✅ Hybrid RAG pipeline initialized successfully!")
#             print(f"🔍 Available methods: semantic, keyword, hybrid, auto")
#             print(f"🏷️  Available filters: company, year, has_financial_data")
#             return True
#         else:
#             print("❌ Failed to build indices")
#             return False


# # Test the hybrid pipeline
# if __name__ == "__main__":
#     print("🧪 Testing Hybrid Financial RAG Pipeline\n")
    
#     # Initialize
#     rag = HybridFinancialRAG()
    
#     if not rag.initialize():
#         print("❌ Failed to initialize")
#         exit(1)
    
#     # Test different search methods
#     test_queries = [
#         "Microsoft cloud revenue 2023",  # Should auto-select hybrid (no longer triggers keyword)
#         "margin 64.9 %",        # Should auto-select keyword  
#         "AI strategy investments",       # Should auto-select hybrid
#         "NVDA revenue growth 2024",      # Should auto-select hybrid (no longer triggers keyword)
#         "MSFT 2023",                     # Should auto-select keyword (company + year together)
#     ]
    
#     print("\n" + "="*70)
#     print("🔍 TESTING SMART RETRIEVAL (Auto Method Selection)")
#     print("="*70)
    
#     for query in test_queries:
#         print(f"\n❓ Query: {query}")
#         print("-" * 60)
        
#         results = rag.smart_retrieve(query, top_k=3)
        
#         for i, result in enumerate(results, 1):
#             print(f"{i}. [{result.company} {result.year}] Method: {result.retrieval_method}")
#             print(f"   Scores: Semantic={result.similarity_score:.3f}, BM25={result.bm25_score:.3f}, Combined={result.combined_score:.3f}")
#             print(f"   Financial: {result.mention_count} mentions, Entities: {result.entity_references[:3]}")
#             print(f"   Content: {result.content[:150]}...")
    
#     print(f"\n✅ Hybrid RAG testing complete!")
