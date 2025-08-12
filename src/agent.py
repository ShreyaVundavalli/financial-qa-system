import re
import json
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class QueryResult:
    query: str
    chunks: List[Any]
    company: Optional[str] = None
    year: Optional[str] = None
    metric: Optional[str] = None

class FinancialQueryAgent:
    def __init__(self, rag_pipeline):
        """Initialize agent with the improved RAG pipeline"""
        self.rag = rag_pipeline
        
        # Company mappings for query parsing
        self.company_aliases = {
            'microsoft': 'MSFT', 'msft': 'MSFT', 'ms': 'MSFT',
            'google': 'GOOGL', 'googl': 'GOOGL', 'alphabet': 'GOOGL', 
            'nvidia': 'NVDA', 'nvda': 'NVDA'
        }
        
        # Enhanced metrics including qualitative terms
        self.metric_aliases = {
            'total revenue': 'revenue', 'revenues': 'revenue', 'sales': 'revenue',
            'operating margin': 'operating margin', 'operating income': 'operating income',
            'gross margin': 'gross margin', 'gross profit': 'gross profit',
            'net income': 'net income', 'profit': 'net income',
            'r&d': 'research and development', 'research': 'research and development',
            'cloud revenue': 'cloud revenue', 'data center': 'data center revenue',
            'ai': 'artificial intelligence', 'advertising': 'advertising revenue',
            # Qualitative terms
            'risks': 'risk factors', 'risk factors': 'risk factors',
            'ai risks': 'ai risks', 'strategy': 'business strategy',
            'competition': 'competitive risks', 'regulation': 'regulatory risks'
        }

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query parsing including qualitative analysis detection"""
        query_lower = query.lower()
        
        # Extract companies
        companies = []
        for alias, standard in self.company_aliases.items():
            if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
                if standard not in companies:
                    companies.append(standard)
        
        # Enhanced multi-company detection
        multi_company_indicators = [
            'compare', 'which', 'highest', 'lowest', 'across', 'between', 
            'all three', 'each company', 'three companies', 'companies'
        ]
        
        if not companies and any(indicator in query_lower for indicator in multi_company_indicators):
            companies = ['MSFT', 'GOOGL', 'NVDA']
        elif 'each company' in query_lower or 'three companies' in query_lower:
            companies = ['MSFT', 'GOOGL', 'NVDA']
        
        # Extract years
        years = list(set(re.findall(r'(202[0-9])', query)))
        
        # Handle year ranges
        year_range_match = re.search(r'from\s+(202[0-9])\s+to\s+(202[0-9])', query_lower)
        if year_range_match:
            years = [year_range_match.group(1), year_range_match.group(2)]
        
        # Extract metrics
        metrics = []
        for alias, standard in self.metric_aliases.items():
            if alias in query_lower:
                if standard not in metrics:
                    metrics.append(standard)
        
        # Detect qualitative vs quantitative queries
        qualitative_indicators = [
            'risks', 'strategy', 'mentioned', 'differ', 'discuss', 'describe',
            'challenges', 'opportunities', 'competitive', 'regulatory'
        ]
        
        is_qualitative = any(indicator in query_lower for indicator in qualitative_indicators)
        
        # Enhanced query type determination
        query_type = self._determine_query_type(query_lower, companies, years, metrics, is_qualitative)
        
        return {
            'original_query': query,
            'companies': companies,
            'years': years,
            'metrics': metrics,
            'query_type': query_type,
            'is_qualitative': is_qualitative,
            'contains_comparison': any(word in query_lower for word in multi_company_indicators),
            'contains_growth': any(word in query_lower for word in ['grow', 'growth', 'increase', 'change']),
            'contains_percentage': 'percentage' in query_lower or '%' in query
        }

    def _determine_query_type(self, query_lower: str, companies: List[str], years: List[str], metrics: List[str], is_qualitative: bool) -> str:
        """Enhanced query type determination including qualitative queries"""
        
        # Qualitative analysis queries
        if is_qualitative:
            if len(companies) > 1 or 'each company' in query_lower or 'compare' in query_lower:
                return 'qualitative_comparison'
            else:
                return 'qualitative_simple'
        
        # Complex multi-aspect indicators (highest priority for quantitative)
        complex_indicators = [
            'compare.*across.*companies', 'each company.*change', 'all three companies',
            'growth rates.*across', 'compare.*all three', 'each company.*margin',
            'percentage.*across.*companies', 'r&d.*percentage.*revenue'
        ]
        
        if any(re.search(pattern, query_lower) for pattern in complex_indicators):
            return 'complex_multi_aspect'
        
        # Rest of quantitative query types...
        elif (len(companies) > 1 and 
              any(word in query_lower for word in ['which', 'highest', 'lowest', 'most', 'least'])):
            return 'cross_company'
        elif (len(companies) > 1 and len(years) > 1) or 'compare' in query_lower:
            return 'complex_multi_aspect'
        elif (len(companies) == 1 and len(years) > 1 and 
              any(word in query_lower for word in ['grow', 'growth', 'change', 'from', 'to'])):
            return 'yoy_comparison'
        elif 'percentage' in query_lower or '%' in query_lower:
            return 'segment_analysis'
        else:
            return 'simple_direct'

    def generate_sub_queries(self, parsed_query: Dict[str, Any]) -> List[str]:
        """Enhanced sub-query generation for both quantitative and qualitative queries"""
        query_type = parsed_query['query_type']
        companies = parsed_query['companies'] or ['MSFT', 'GOOGL', 'NVDA']
        years = parsed_query['years'] or ['2023', '2024']  # Default to recent years
        metrics = parsed_query['metrics'] or ['revenue']
        
        sub_queries = []
        
        # Qualitative query handling
        if query_type in ['qualitative_comparison', 'qualitative_simple']:
            
            # AI risks specific queries
            if 'ai' in parsed_query['original_query'].lower() and 'risk' in parsed_query['original_query'].lower():
                for company in companies:
                    for year in years[:2]:  # Recent years
                        sub_queries.extend([
                            f"{company} artificial intelligence risks {year}",
                            f"{company} AI challenges {year}",
                            f"{company} machine learning risks {year}",
                            f"{company} technology risks {year}"
                        ])
            
            # General risk factors
            elif 'risk' in parsed_query['original_query'].lower():
                for company in companies:
                    for year in years[:2]:
                        sub_queries.extend([
                            f"{company} risk factors {year}",
                            f"{company} business risks {year}",
                            f"{company} competitive risks {year}"
                        ])
            
            # Strategy queries
            elif 'strategy' in parsed_query['original_query'].lower():
                for company in companies:
                    for year in years[:2]:
                        sub_queries.extend([
                            f"{company} business strategy {year}",
                            f"{company} competitive strategy {year}",
                            f"{company} technology strategy {year}"
                        ])
            
            # Fallback for other qualitative queries
            else:
                for company in companies:
                    for year in years[:2]:
                        for metric in metrics[:1]:
                            sub_queries.append(f"{company} {metric} {year}")
        
        # Quantitative query handling (same as before)
        elif query_type == 'simple_direct':
            company = companies[0] if companies else 'MSFT'
            year = years[0] if years else '2023'
            metric = metrics[0] if metrics else 'revenue'
            sub_queries.append(f"{company} {metric} {year}")
            
        elif query_type == 'yoy_comparison':
            company = companies[0] if companies else 'MSFT'
            metric = metrics[0] if metrics else 'revenue'
            for year in years:
                sub_queries.append(f"{company} {metric} {year}")
                
        elif query_type == 'cross_company':
            year = years[0] if years else '2023'
            metric = metrics[0] if metrics else 'revenue'
            for company in companies:
                sub_queries.append(f"{company} {metric} {year}")
                
        elif query_type == 'segment_analysis':
            company = companies[0] if companies else 'GOOGL'
            year = years[0] if years else '2023'
            sub_queries.extend([
                f"{company} total revenue {year}",
                f"{company} cloud revenue {year}",
                f"{company} advertising revenue {year}"
            ])
            
        else:  # complex_multi_aspect
            # R&D percentage special case
            if ('r&d' in parsed_query['original_query'].lower() and 'percentage' in parsed_query['original_query'].lower()):
                year = years[0] if years else '2023'
                for company in companies:
                    sub_queries.extend([
                        f"{company} research and development {year}",
                        f"{company} total revenue {year}"
                    ])
            # Other complex cases...
            else:
                for company in companies[:3]:
                    for year in years[:2]:
                        for metric in metrics[:1]:
                            sub_queries.append(f"{company} {metric} {year}")
        
        return sub_queries[:15]  # Increased limit for comprehensive analysis

    def execute_sub_queries(self, sub_queries: List[str]) -> List[QueryResult]:
        """Execute all sub-queries and collect results"""
        results = []
        
        for sub_query in sub_queries:
            print(f"  🔍 Executing: {sub_query}")
            
            # Use smart retrieval from your improved RAG pipeline
            chunks = self.rag.smart_retrieve(sub_query, top_k=3)
            
            # Extract company and year from sub-query for metadata
            company_match = re.search(r'\b(MSFT|GOOGL|NVDA)\b', sub_query)
            year_match = re.search(r'\b(202[0-9])\b', sub_query)
            
            result = QueryResult(
                query=sub_query,
                chunks=chunks,
                company=company_match.group(1) if company_match else None,
                year=year_match.group(1) if year_match else None
            )
            results.append(result)
        
        return results

    def _synthesize_qualitative_comparison(self, parsed_query: Dict[str, Any], sub_results: List[QueryResult]) -> Dict[str, Any]:
        """ENHANCED: AI risks analysis with fallback to financial data insights"""
        company_insights = {}
        sources = []
        
        # First pass: Look for any meaningful AI-related content
        for result in sub_results:
            if result.chunks and result.company:
                for chunk in result.chunks:
                    content_lower = chunk.content.lower()
                    content = chunk.content
                    
                    # Enhanced AI detection (very broad)
                    ai_indicators = [
                        'artificial intelligence', 'ai technology', 'ai ', 'machine learning',
                        'intelligent', 'automation', 'algorithm', 'data analytics',
                        'cloud computing', 'platform', 'digital', 'innovation',
                        'technology', 'software', 'computing', 'analytics'
                    ]
                    
                    # Check if chunk has meaningful content (not just XML tags)
                    is_meaningful = (
                        len(content) > 100 and
                        not content.startswith('BusinessProcessesMember') and
                        ' ' in content and  # Has spaces (not just codes)
                        any(word in content_lower for word in ['the', 'and', 'our', 'we', 'company'])
                    )
                    
                    # Look for AI-related content or technology discussions
                    has_tech_content = any(term in content_lower for term in ai_indicators)
                    
                    if is_meaningful and has_tech_content:
                        if result.company not in company_insights:
                            company_insights[result.company] = []
                        
                        # Extract meaningful sentences
                        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
                        tech_sentences = []
                        
                        for sentence in sentences:
                            sentence_lower = sentence.lower()
                            # Look for business/technology discussion
                            if (any(term in sentence_lower for term in ai_indicators[:8]) or  # Core AI terms
                                (any(term in sentence_lower for term in ['business', 'strategy', 'product', 'service', 'market']) and
                                any(term in sentence_lower for term in ['technology', 'digital', 'innovation', 'platform']))):
                                tech_sentences.append(sentence)
                        
                        if tech_sentences:
                            company_insights[result.company].extend(tech_sentences[:2])
                            sources.append({
                                'company': result.company,
                                'year': chunk.year,
                                'excerpt': content[:400],
                                'value': getattr(chunk, 'value', 0),
                                'type': chunk.get('type', 'unknown')
                            })
        
        # Fallback: Use financial data to infer AI investment/strategy
        if not company_insights:
            print("  📊 No narrative AI content found, using financial indicators...")
            
            # Look for R&D expenses and technology-related financial metrics
            for result in sub_results:
                if result.chunks and result.company:
                    for chunk in result.chunks:
                        content_lower = chunk.content.lower()
                        
                        # Look for technology-related financial metrics
                        if any(term in content_lower for term in [
                            'research and development', 'r&d', 'technology', 'software',
                            'cloud', 'data center', 'artificial intelligence', 'platform'
                        ]):
                            if result.company not in company_insights:
                                company_insights[result.company] = []
                            
                            # Create insight from financial data
                            if hasattr(chunk, 'value') and chunk.value > 0:
                                insight = f"Financial filings show significant technology investments with {chunk.content.split(':')[1].strip() if ':' in chunk.content else 'technology-related expenses'}"
                                company_insights[result.company].append(insight)
                                
                                sources.append({
                                    'company': result.company,
                                    'year': chunk.year,
                                    'excerpt': chunk.content,
                                    'value': getattr(chunk, 'value', 0)
                                })
                                break  # One per company is enough
        
        # Generate response based on what we found
        if not company_insights:
            return {
                'query': parsed_query['original_query'],
                'answer': "Limited AI-specific information available in the current dataset. The system searched through available SEC filing data but found insufficient narrative content about AI risks and strategies. This type of analysis would benefit from access to complete Risk Factors and Business Strategy sections of 10-K filings.",
                'reasoning': "Searched through available financial and narrative data but found limited AI-specific content in the indexed sections.",
                'sub_queries': [result.query for result in sub_results],
                'sources': sources[:3],
                'confidence': 0.3
            }
        
        # Generate comparative analysis
        comparison_parts = []
        total_insights = 0
        
        for company, insights in company_insights.items():
            if insights:
                unique_insights = list(set(insights))[:1]  # One key insight per company
                total_insights += len(unique_insights)
                
                # Determine company's AI/tech focus
                company_focus = "technology and innovation investments"
                insight_text = unique_insights[0] if unique_insights else ""
                
                # Identify specific focus areas
                insight_lower = insight_text.lower()
                if 'cloud' in insight_lower or 'platform' in insight_lower:
                    company_focus = "cloud platform and infrastructure technology"
                elif 'artificial intelligence' in insight_lower or 'ai' in insight_lower:
                    company_focus = "artificial intelligence and machine learning"
                elif 'research' in insight_lower or 'development' in insight_lower:
                    company_focus = "research and development in emerging technologies"
                
                comparison_parts.append(f"{company}: Focuses on {company_focus}")
        
        # Create comprehensive answer
        if comparison_parts:
            answer = f"AI and technology strategy comparison based on available data: {'; '.join(comparison_parts)}. Analysis indicates different technological focus areas across companies, though detailed AI risk discussions would require access to complete Risk Factors sections."
            confidence = min(0.85, 0.6 + (len(company_insights) * 0.1))
        else:
            answer = "No comprehensive AI risk analysis available in current dataset."
            confidence = 0.4
        
        return {
            'query': parsed_query['original_query'],
            'answer': answer,
            'reasoning': f"Analyzed available financial and narrative data across {len(company_insights)} companies for technology and AI-related information. Limited by availability of comprehensive narrative sections in current dataset.",
            'sub_queries': [result.query for result in sub_results],
            'sources': sources[:6],
            'confidence': confidence,
            'qualitative_insights': {comp: len(insights) for comp, insights in company_insights.items()},
            'analysis_note': "Analysis limited by data quality in narrative sections. Future improvements would benefit from better narrative text extraction from 10-K filings."
        }



    def synthesize_answer(self, parsed_query: Dict[str, Any], sub_results: List[QueryResult]) -> Dict[str, Any]:
        """Enhanced synthesis including qualitative analysis"""
        query_type = parsed_query['query_type']
        
        # Handle qualitative queries
        if query_type in ['qualitative_comparison', 'qualitative_simple']:
            return self._synthesize_qualitative_comparison(parsed_query, sub_results)
        
        # Handle quantitative queries (same as before)
        elif query_type == 'simple_direct':
            return self._synthesize_simple_answer(parsed_query, sub_results)
        elif query_type == 'yoy_comparison':
            return self._synthesize_growth_answer(parsed_query, sub_results)
        elif query_type == 'cross_company':
            return self._synthesize_comparison_answer(parsed_query, sub_results)
        elif query_type == 'segment_analysis':
            return self._synthesize_segment_answer(parsed_query, sub_results)
        else:  # complex_multi_aspect
            return self._synthesize_complex_answer(parsed_query, sub_results)

    # Include all other methods from the previous version (unchanged)
    def extract_financial_values(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """Extract numerical values from chunks"""
        values = []
        
        for chunk in chunks:
            value_info = {
                'company': chunk.company,
                'year': chunk.year,
                'value': getattr(chunk, 'value', 0),
                'content': chunk.content,
                'metric': getattr(chunk, 'metric', 'unknown')
            }
            values.append(value_info)
        
        return values

    def calculate_growth(self, value1: float, value2: float, year1: str, year2: str) -> Dict[str, Any]:
        """Calculate growth between two values"""
        if value1 == 0:
            return {'growth_rate': None, 'description': 'Cannot calculate growth from zero'}
        
        growth_rate = ((value2 - value1) / value1) * 100
        growth_amount = value2 - value1
        
        return {
            'growth_rate': growth_rate,
            'growth_amount': growth_amount,
            'from_year': year1,
            'to_year': year2,
            'from_value': value1,
            'to_value': value2,
            'description': f"Growth of {growth_rate:.1f}% from {year1} to {year2}"
        }

    def _synthesize_simple_answer(self, parsed_query: Dict[str, Any], sub_results: List[QueryResult]) -> Dict[str, Any]:
        """Synthesize answer for simple direct queries"""
        if not sub_results or not sub_results[0].chunks:
            return self._create_no_data_response(parsed_query)
        
        best_chunk = sub_results[0].chunks[0]
        
        value = getattr(best_chunk, 'value', None)
        if value is None or value <= 0:
            for result in sub_results:
                for chunk in result.chunks:
                    chunk_value = getattr(chunk, 'value', None)
                    if chunk_value and chunk_value > 0:
                        best_chunk = chunk
                        value = chunk_value
                        break
                if value and value > 0:
                    break
        
        if not value or value <= 0:
            return self._create_no_data_response(parsed_query)
        
        company = best_chunk.company
        year = best_chunk.year
        metric = parsed_query['metrics'][0] if parsed_query['metrics'] else 'revenue'
        
        answer = f"{company}'s {metric} in {year} was ${value:,.0f}."
        
        sources = [{
            'company': best_chunk.company,
            'year': best_chunk.year,
            'excerpt': best_chunk.content,
            'value': value
        }]
        
        return {
            'query': parsed_query['original_query'],
            'answer': answer,
            'reasoning': f"Retrieved {metric} for {company} in {year} from SEC filings.",
            'sub_queries': [result.query for result in sub_results],
            'sources': sources,
            'confidence': 1.0
        }

    def _synthesize_growth_answer(self, parsed_query: Dict[str, Any], sub_results: List[QueryResult]) -> Dict[str, Any]:
        """Synthesize answer for year-over-year comparison queries"""
        if len(sub_results) < 2:
            return self._create_no_data_response(parsed_query)
        
        year_values = {}
        sources = []
        
        for result in sub_results:
            if result.chunks:
                chunk = result.chunks[0]
                value = getattr(chunk, 'value', None)
                if value and value > 0:
                    year_values[chunk.year] = value
                    sources.append({
                        'company': chunk.company,
                        'year': chunk.year,
                        'excerpt': chunk.content,
                        'value': value
                    })
        
        if len(year_values) < 2:
            return self._create_no_data_response(parsed_query)
        
        years = sorted(year_values.keys())
        year1, year2 = years[0], years[-1]
        value1, value2 = year_values[year1], year_values[year2]
        
        growth_info = self.calculate_growth(value1, value2, year1, year2)
        
        company = parsed_query['companies'][0] if parsed_query['companies'] else 'Company'
        metric = parsed_query['metrics'][0] if parsed_query['metrics'] else 'revenue'
        
        if growth_info['growth_rate'] is not None:
            answer = f"{company}'s {metric} grew by {growth_info['growth_rate']:.1f}% from {year1} to {year2}, increasing from ${value1:,.0f} to ${value2:,.0f}."
        else:
            answer = f"Unable to calculate growth for {company}'s {metric} due to insufficient data."
        
        return {
            'query': parsed_query['original_query'],
            'answer': answer,
            'reasoning': f"Calculated growth by comparing {metric} values between {year1} and {year2}.",
            'sub_queries': [result.query for result in sub_results],
            'sources': sources,
            'confidence': 1.0 if growth_info['growth_rate'] is not None else 0.5,
            'growth_calculation': growth_info
        }

    def _synthesize_comparison_answer(self, parsed_query: Dict[str, Any], sub_results: List[QueryResult]) -> Dict[str, Any]:
        """Synthesize answer for cross-company comparison queries"""
        company_values = {}
        sources = []
        
        for result in sub_results:
            if result.chunks:
                chunk = result.chunks[0]
                value = getattr(chunk, 'value', None)
                if value and value > 0:
                    company_values[chunk.company] = value
                    sources.append({
                        'company': chunk.company,
                        'year': chunk.year,
                        'excerpt': chunk.content,
                        'value': value
                    })
        
        if not company_values:
            return self._create_no_data_response(parsed_query)
        
        query_lower = parsed_query['original_query'].lower()
        if 'highest' in query_lower or 'most' in query_lower:
            best_company = max(company_values.keys(), key=lambda x: company_values[x])
            comparison_type = 'highest'
        else:
            best_company = min(company_values.keys(), key=lambda x: company_values[x])
            comparison_type = 'lowest'
        
        metric = parsed_query['metrics'][0] if parsed_query['metrics'] else 'revenue'
        year = parsed_query['years'][0] if parsed_query['years'] else '2023'
        
        comparison_details = []
        for company, value in sorted(company_values.items(), key=lambda x: x[1], reverse=(comparison_type == 'highest')):
            comparison_details.append(f"{company}: ${value:,.0f}")
        
        answer = f"{best_company} had the {comparison_type} {metric} in {year} at ${company_values[best_company]:,.0f}. "
        answer += f"Comparison: {', '.join(comparison_details)}."
        
        return {
            'query': parsed_query['original_query'],
            'answer': answer,
            'reasoning': f"Compared {metric} across {len(company_values)} companies for {year} to identify the {comparison_type} value.",
            'sub_queries': [result.query for result in sub_results],
            'sources': sources,
            'confidence': 1.0,
            'comparison_data': company_values
        }

    def _synthesize_segment_answer(self, parsed_query: Dict[str, Any], sub_results: List[QueryResult]) -> Dict[str, Any]:
        """Synthesize answer for segment analysis queries (percentage calculations)"""
        total_revenue = None
        segment_revenue = None
        sources = []
        
        for result in sub_results:
            if result.chunks:
                chunk = result.chunks[0]
                value = getattr(chunk, 'value', None)
                if value and value > 0:
                    if 'total' in result.query.lower():
                        total_revenue = value
                    elif any(seg in result.query.lower() for seg in ['cloud', 'advertising', 'data center']):
                        segment_revenue = value
                    
                    sources.append({
                        'company': chunk.company,
                        'year': chunk.year,
                        'excerpt': chunk.content,
                        'value': value
                    })
        
        if total_revenue and segment_revenue:
            percentage = (segment_revenue / total_revenue) * 100
            company = parsed_query['companies'][0] if parsed_query['companies'] else 'Company'
            segment = 'segment'
            if 'cloud' in parsed_query['original_query'].lower():
                segment = 'cloud'
            elif 'advertising' in parsed_query['original_query'].lower():
                segment = 'advertising'
            
            answer = f"{company}'s {segment} revenue represented {percentage:.1f}% of total revenue, with {segment} revenue of ${segment_revenue:,.0f} out of total revenue of ${total_revenue:,.0f}."
            
            return {
                'query': parsed_query['original_query'],
                'answer': answer,
                'reasoning': f"Calculated percentage by dividing {segment} revenue by total revenue.",
                'sub_queries': [result.query for result in sub_results],
                'sources': sources,
                'confidence': 1.0,
                'calculation': {'total': total_revenue, 'segment': segment_revenue, 'percentage': percentage}
            }
        
        return self._create_no_data_response(parsed_query)

    def _synthesize_complex_answer(self, parsed_query: Dict[str, Any], sub_results: List[QueryResult]) -> Dict[str, Any]:
        """Enhanced synthesis for complex multi-aspect queries"""
        query_lower = parsed_query['original_query'].lower()
        
        # Handle R&D percentage comparison
        if 'r&d' in query_lower and 'percentage' in query_lower:
            return self._synthesize_rd_percentage_comparison(parsed_query, sub_results)
        
        # Handle growth comparison queries
        elif ('growth' in query_lower or 'change' in query_lower) and len(parsed_query['companies']) > 1:
            return self._synthesize_multi_company_growth(parsed_query, sub_results)
        
        # Default complex analysis
        else:
            return self._synthesize_general_complex(parsed_query, sub_results)

    def _synthesize_rd_percentage_comparison(self, parsed_query: Dict[str, Any], sub_results: List[QueryResult]) -> Dict[str, Any]:
        """Fixed: R&D as percentage of revenue comparison with proper calculation"""
        company_data = {}
        sources = []
        
        # Group results by company
        for result in sub_results:
            if result.chunks and result.company:
                if result.company not in company_data:
                    company_data[result.company] = {}
                
                chunk = result.chunks[0]
                value = getattr(chunk, 'value', None)
                if value and value > 0:
                    # Determine if this is R&D or revenue based on query
                    if 'research' in result.query.lower() or 'r&d' in result.query.lower():
                        company_data[result.company]['rd'] = value
                    elif 'revenue' in result.query.lower():
                        company_data[result.company]['revenue'] = value
                    
                    sources.append({
                        'company': result.company,
                        'year': chunk.year,
                        'excerpt': chunk.content,
                        'value': value
                    })
        
        # Calculate percentages
        company_percentages = {}
        comparison_details = []
        
        for company, data in company_data.items():
            if 'rd' in data and 'revenue' in data:
                rd_value = data['rd']
                revenue_value = data['revenue']
                percentage = (rd_value / revenue_value) * 100
                
                company_percentages[company] = {
                    'rd': rd_value,
                    'revenue': revenue_value,
                    'percentage': percentage
                }
                
                comparison_details.append(f"{company}: {percentage:.1f}% (${rd_value:,.0f} R&D / ${revenue_value:,.0f} revenue)")
        
        if not company_percentages:
            # Fallback to absolute R&D comparison if percentage calculation fails
            rd_only_data = {}
            for company, data in company_data.items():
                if 'rd' in data:
                    rd_only_data[company] = data['rd']
            
            if rd_only_data:
                sorted_companies = sorted(rd_only_data.items(), key=lambda x: x[1], reverse=True)
                comparison_details = [f"{company}: ${value:,.0f}" for company, value in sorted_companies]
                
                answer = f"R&D spending comparison in 2023: {', '.join(comparison_details)}. Note: Percentage calculations require both R&D and total revenue data for each company."
                
                return {
                    'query': parsed_query['original_query'],
                    'answer': answer,
                    'reasoning': "Compared absolute R&D spending. Percentage calculations would require matching total revenue data for each company.",
                    'sub_queries': [result.query for result in sub_results],
                    'sources': sources,
                    'confidence': 0.7,
                    'comparison_data': rd_only_data
                }
            else:
                return self._create_no_data_response(parsed_query)
        
        # Sort by percentage
        sorted_percentages = sorted(company_percentages.items(), key=lambda x: x[1]['percentage'], reverse=True)
        highest_company = sorted_percentages[0][0]
        highest_percentage = sorted_percentages[0][1]['percentage']
        
        answer = f"R&D spending as percentage of revenue in 2023: {', '.join(comparison_details)}. {highest_company} had the highest R&D intensity at {highest_percentage:.1f}%."
        
        return {
            'query': parsed_query['original_query'],
            'answer': answer,
            'reasoning': f"Calculated R&D as percentage of total revenue for each company and compared across {len(company_percentages)} companies.",
            'sub_queries': [result.query for result in sub_results],
            'sources': sources,
            'confidence': 1.0,
            'percentage_comparison': {comp: data['percentage'] for comp, data in company_percentages.items()}
        }

    def _synthesize_multi_company_growth(self, parsed_query: Dict[str, Any], sub_results: List[QueryResult]) -> Dict[str, Any]:
        """Synthesize multi-company growth comparison"""
        companies = parsed_query['companies']
        years = sorted(parsed_query['years']) if parsed_query['years'] else ['2022', '2024']
        
        if len(years) < 2:
            return self._create_no_data_response(parsed_query)
        
        year1, year2 = years[0], years[-1]
        company_growth = {}
        sources = []
        
        # Group results by company
        company_data = {}
        for result in sub_results:
            if result.chunks and result.company:
                if result.company not in company_data:
                    company_data[result.company] = {}
                
                chunk = result.chunks[0]
                value = getattr(chunk, 'value', None)
                if value and value > 0:
                    company_data[result.company][chunk.year] = {
                        'value': value,
                        'chunk': chunk
                    }
        
        # Calculate growth for each company
        growth_results = []
        for company, year_data in company_data.items():
            if year1 in year_data and year2 in year_data:
                val1 = year_data[year1]['value']
                val2 = year_data[year2]['value']
                
                growth_calc = self.calculate_growth(val1, val2, year1, year2)
                company_growth[company] = growth_calc['growth_rate']
                
                growth_results.append(f"{company}: {growth_calc['growth_rate']:.1f}% growth")
                
                # Add sources
                sources.extend([
                    {
                        'company': company,
                        'year': year1,
                        'excerpt': year_data[year1]['chunk'].content,
                        'value': val1
                    },
                    {
                        'company': company,
                        'year': year2,
                        'excerpt': year_data[year2]['chunk'].content,
                        'value': val2
                    }
                ])
        
        if not growth_results:
            return self._create_no_data_response(parsed_query)
        
        metric = parsed_query['metrics'][0] if parsed_query['metrics'] else 'operating margin'
        answer = f"Comparison of {metric} change from {year1} to {year2}: {', '.join(growth_results)}."
        
        return {
            'query': parsed_query['original_query'],
            'answer': answer,
            'reasoning': f"Calculated {metric} growth for each company between {year1} and {year2}.",
            'sub_queries': [result.query for result in sub_results],
            'sources': sources,
            'confidence': 1.0,
            'growth_comparison': company_growth
        }

    def _synthesize_general_complex(self, parsed_query: Dict[str, Any], sub_results: List[QueryResult]) -> Dict[str, Any]:
        """General complex query synthesis"""
        data_points = []
        sources = []
        
        for result in sub_results:
            if result.chunks:
                chunk = result.chunks[0]
                value = getattr(chunk, 'value', None)
                if value and value > 0:
                    data_points.append(f"{chunk.company} ({chunk.year}): ${value:,.0f}")
                    sources.append({
                        'company': chunk.company,
                        'year': chunk.year,
                        'excerpt': chunk.content,
                        'value': value
                    })
        
        if not data_points:
            return self._create_no_data_response(parsed_query)
        
        answer = f"Financial analysis results: {'; '.join(data_points[:6])}."  # Limit to top 6
        
        return {
            'query': parsed_query['original_query'],
            'answer': answer,
            'reasoning': f"Analyzed multiple financial metrics across companies and years.",
            'sub_queries': [result.query for result in sub_results],
            'sources': sources[:8],  # Limit sources
            'confidence': 0.8
        }

    def _create_no_data_response(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Create response when no relevant data is found"""
        return {
            'query': parsed_query['original_query'],
            'answer': "Unable to find relevant data for this query in the available SEC filings.",
            'reasoning': "No matching information found in the available SEC filing data.",
            'sub_queries': [],
            'sources': [],
            'confidence': 0.0
        }

    def run_query(self, query: str) -> Dict[str, Any]:
        """Main entry point for processing any query"""
        print(f"🤖 Processing query: {query}")
        
        parsed_query = self.parse_query(query)
        print(f"📊 Query type: {parsed_query['query_type']}")
        print(f"📊 Companies: {parsed_query['companies']}")
        print(f"📊 Years: {parsed_query['years']}")
        print(f"📊 Metrics: {parsed_query['metrics']}")
        print(f"📊 Is qualitative: {parsed_query['is_qualitative']}")
        
        sub_queries = self.generate_sub_queries(parsed_query)
        print(f"🔍 Generated {len(sub_queries)} sub-queries")
        
        sub_results = self.execute_sub_queries(sub_queries)
        
        final_response = self.synthesize_answer(parsed_query, sub_results)
        
        print(f"✅ Generated response with confidence: {final_response['confidence']}")
        return final_response


