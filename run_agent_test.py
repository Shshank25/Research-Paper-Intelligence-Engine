import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.agent import ResearchAgent

if __name__ == '__main__':
    topic = "Retrieval-Augmented Generation in Healthcare"
    print(f"==================================================")
    print(f"Running 7th-Semester Agentic Workflow Test")
    print(f"Topic: {topic}")
    print(f"==================================================\n")
    
    agent = ResearchAgent()
    result = agent.run(topic)
    
    discovered = result.get("discovered_papers", [])
    selected = result.get("selected_papers", [])
    analyses = result.get("paper_analysis", [])
    
    print("\n==================================================")
    print("WORKFLOW TEST RESULTS METRICS")
    print("==================================================")
    print(f"1. Papers Discovered    : {len(discovered)}")
    print(f"2. Papers Selected      : {len(selected)}")
    print(f"3. Papers Analyzed      : {len(analyses)}")
    print("4. Paper Ranking Scores :")
    for p in selected:
        print(f"   - [{p.get('rank', 1)}] {p.get('title', 'Unknown')[:50]}... | Score: {p.get('relevance_score', 0.0)}")
        
    print("\n--------------------------------------------------")
    print("5. Multi-Paper Comparison Matrix:")
    print("--------------------------------------------------")
    print(result.get("comparison"))
    
    print("\n--------------------------------------------------")
    print("6. Literature Review:")
    print("--------------------------------------------------")
    print(result.get("literature_review"))
    
    print("\n==================================================")
    print("✅ TEST WORKFLOW COMPLETE!")
    print("==================================================")


