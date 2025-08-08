# enhanced_diagnostic_check.py
import json
from collections import Counter

# Load and inspect the chunks
with open('data/rag_chunks.json', 'r') as f:
    chunks = json.load(f)

# Analyze narrative chunks
narrative_chunks = [c for c in chunks if c.get('type') == 'narrative_text']
print(f"Total narrative chunks: {len(narrative_chunks)}")

# Check distribution by company
company_dist = Counter(c['company'] for c in narrative_chunks)
print(f"\nNarrative chunks by company:")
for company, count in company_dist.items():
    print(f"  {company}: {count} chunks")

# Check for AI content with expanded search terms
ai_terms = [
    'artificial intelligence', 'ai ', ' ai,', ' ai.', 'ai technology', 'ai systems',
    'machine learning', 'deep learning', 'neural networks',
    'algorithms', 'automation', 'intelligent', 'generative',
    'openai', 'chatgpt', 'gpt', 'llm', 'large language model',
    'copilot', 'assistant', 'chatbot'
]

ai_chunks = []
for chunk in narrative_chunks:
    content_lower = chunk['content'].lower()
    if any(ai_term in content_lower for ai_term in ai_terms):
        ai_chunks.append(chunk)

print(f"\nAI-related narrative chunks: {len(ai_chunks)}")

# Show distribution by company
ai_company_dist = Counter(c['company'] for c in ai_chunks)
print(f"\nAI chunks by company:")
for company, count in ai_company_dist.items():
    print(f"  {company}: {count} chunks")

# Show sample from each company
print(f"\nSample AI chunks by company:")
for company in ['MSFT', 'GOOGL', 'NVDA']:
    company_ai_chunks = [c for c in ai_chunks if c['company'] == company]
    if company_ai_chunks:
        print(f"\n{company} samples:")
        for i, chunk in enumerate(company_ai_chunks[:2]):
            print(f"  {i+1}. [{chunk['company']} {chunk['year']} - {chunk['metric']}]")
            print(f"     {chunk['content'][:150]}...")
    else:
        print(f"\n{company}: No AI-related chunks found")

# Check what sections were extracted for each company
print(f"\nNarrative sections by company:")
for company in ['MSFT', 'GOOGL', 'NVDA']:
    company_sections = {}
    for chunk in narrative_chunks:
        if chunk['company'] == company:
            metric = chunk.get('metric', 'unknown')
            company_sections[metric] = company_sections.get(metric, 0) + 1
    
    print(f"  {company}: {dict(company_sections)}")
