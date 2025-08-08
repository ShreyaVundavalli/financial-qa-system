from rag_pipeline import SECFinancialRAG
from agent import FinancialQueryAgent
import json
import sys


def interactive_mode():
    """Allow users to ask custom queries"""
    print("🤖 Financial Q&A System - Interactive Mode")
    print("Ask questions about MSFT, GOOGL, or NVDA financial data (2022-2024)")
    print("Type 'quit' to exit\n")
    
    # Initialize system
    rag = SECFinancialRAG()
    if not rag.initialize():
        print("❌ Failed to initialize system")
        return
    
    agent = FinancialQueryAgent(rag)
    
    while True:
        user_query = input("\n💬 Your question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if len(user_query) < 5:
            print("⚠️  Please ask a more detailed question")
            continue
        
        try:
            print(f"\n🔍 Processing: {user_query}")
            response = agent.run_query(user_query)
            
            print(f"\n📊 Answer: {response['answer']}")
            print(f"🎯 Confidence: {response['confidence']}")
            
            if response['confidence'] < 0.7:
                print("💡 Note: Low confidence - answer may be limited by available data")
            
            # Show reasoning if available
            if 'reasoning' in response:
                print(f"🧠 Reasoning: {response['reasoning']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    print("🚀 Comprehensive Financial Q&A System Testing\n")
    
    # Initialize RAG pipeline
    rag = SECFinancialRAG()
    if not rag.initialize():
        print("❌ Failed to initialize RAG pipeline")
        return
    
    # Initialize agent
    agent = FinancialQueryAgent(rag)
    
    # Test all required query types from assignment
    test_queries = [
        # 1. Basic Metrics
        "What was Microsoft's total revenue in 2023?",
        "What was NVIDIA's total revenue in fiscal year 2024?",
        
        # 2. YoY Comparison  
        "How did NVIDIA's data center revenue grow from 2022 to 2023?",
        "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
        
        # 3. Cross-Company
        "Which company had the highest operating margin in 2023?", 
        "Which of the three companies had the highest gross margin in 2023?",
        
        # 4. Segment Analysis
        "What percentage of Google's 2023 revenue came from advertising?",
        "What percentage of Google's revenue came from cloud in 2023?",
        
        # 5. Complex Multi-aspect
        "Compare the R&D spending as a percentage of revenue across all three companies in 2023",
        "How did each company's operating margin change from 2022 to 2024?",
        "Compare cloud revenue growth rates across all three companies from 2022 to 2023",
        
        # AI Strategy (if data available)
        "What are the main AI risks mentioned by each company and how do they differ?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"🔍 Test Query {i}: {query}")
        print('='*80)
        
        try:
            response = agent.run_query(query)
            results.append({'query': query, 'response': response, 'success': response['confidence'] > 0})
            print(json.dumps(response, indent=2))
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            results.append({'query': query, 'response': None, 'success': False})
    
    # Summary report
    print(f"\n{'='*80}")
    print("📊 TESTING SUMMARY")
    print('='*80)
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"✅ Successful queries: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"❌ Failed queries: {total-successful}/{total}")
    
    # Check coverage of required query types
    query_types = {
        'Basic Metrics': ['What was Microsoft\'s total revenue', 'What was NVIDIA\'s total revenue'],
        'YoY Comparison': ['How did NVIDIA\'s data center revenue grow', 'How much did Microsoft\'s cloud revenue grow'],
        'Cross-Company': ['Which company had the highest operating', 'Which of the three companies had the highest'],
        'Segment Analysis': ['What percentage of Google\'s'],
        'Complex Multi-aspect': ['Compare the R&D spending', 'How did each company\'s operating margin', 'Compare cloud revenue growth']
    }
    
    print(f"\n📋 Query Type Coverage:")
    for qtype, patterns in query_types.items():
        covered = sum(1 for r in results if r['success'] and any(pattern in r['query'] for pattern in patterns))
        total_type = sum(1 for q in test_queries if any(pattern in q for pattern in patterns))
        print(f"   {qtype}: {covered}/{total_type} queries successful")
    
    return results


if __name__ == "__main__":
    # Check if interactive mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        # Run comprehensive tests
        main()



# from rag_pipeline import SECFinancialRAG
# from agent import FinancialQueryAgent
# import json

# def main():
#     print("🚀 Comprehensive Financial Q&A System Testing\n")
    
#     # Initialize RAG pipeline
#     rag = SECFinancialRAG()
#     if not rag.initialize():
#         print("❌ Failed to initialize RAG pipeline")
#         return
    
#     # Initialize agent
#     agent = FinancialQueryAgent(rag)
    
#     # Test all required query types from assignment
#     test_queries = [
#         # 1. Basic Metrics
#         "What was Microsoft's total revenue in 2023?",
#         "What was NVIDIA's total revenue in fiscal year 2024?",
        
#         # 2. YoY Comparison  
#         "How did NVIDIA's data center revenue grow from 2022 to 2023?",
#         "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
        
#         # 3. Cross-Company
#         "Which company had the highest operating margin in 2023?", 
#         "Which of the three companies had the highest gross margin in 2023?",
        
#         # 4. Segment Analysis
#         "What percentage of Google's 2023 revenue came from advertising?",
#         "What percentage of Google's revenue came from cloud in 2023?",
        
#         # 5. Complex Multi-aspect
#         "Compare the R&D spending as a percentage of revenue across all three companies in 2023",
#         "How did each company's operating margin change from 2022 to 2024?",
#         "Compare cloud revenue growth rates across all three companies from 2022 to 2023",
        
#         # AI Strategy (if data available)
#         "What are the main AI risks mentioned by each company and how do they differ?"
#     ]
    
#     results = []
    
#     for i, query in enumerate(test_queries, 1):
#         print(f"\n{'='*80}")
#         print(f"🔍 Test Query {i}: {query}")
#         print('='*80)
        
#         try:
#             response = agent.run_query(query)
#             results.append({'query': query, 'response': response, 'success': response['confidence'] > 0})
#             print(json.dumps(response, indent=2))
#         except Exception as e:
#             print(f"❌ Error processing query: {e}")
#             results.append({'query': query, 'response': None, 'success': False})
    
#     # Summary report
#     print(f"\n{'='*80}")
#     print("📊 TESTING SUMMARY")
#     print('='*80)
    
#     successful = sum(1 for r in results if r['success'])
#     total = len(results)
    
#     print(f"✅ Successful queries: {successful}/{total} ({successful/total*100:.1f}%)")
#     print(f"❌ Failed queries: {total-successful}/{total}")
    
#     # Check coverage of required query types
#     query_types = {
#         'Basic Metrics': ['What was Microsoft\'s total revenue', 'What was NVIDIA\'s total revenue'],
#         'YoY Comparison': ['How did NVIDIA\'s data center revenue grow', 'How much did Microsoft\'s cloud revenue grow'],
#         'Cross-Company': ['Which company had the highest operating', 'Which of the three companies had the highest'],
#         'Segment Analysis': ['What percentage of Google\'s'],
#         'Complex Multi-aspect': ['Compare the R&D spending', 'How did each company\'s operating margin', 'Compare cloud revenue growth']
#     }
    
#     print(f"\n📋 Query Type Coverage:")
#     for qtype, patterns in query_types.items():
#         covered = sum(1 for r in results if r['success'] and any(pattern in r['query'] for pattern in patterns))
#         total_type = sum(1 for q in test_queries if any(pattern in q for pattern in patterns))
#         print(f"   {qtype}: {covered}/{total_type} queries successful")
    
#     return results

# if __name__ == "__main__":
#     main()
