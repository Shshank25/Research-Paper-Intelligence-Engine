import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.agent import ResearchAgent

if __name__ == '__main__':
    topic = "Federated Learning for Privacy-Preserving Medical Image Analysis"
    print(f"Running ResearchAgent on topic: {topic}\n")
    agent = ResearchAgent()
    result = agent.run(topic)
    print("\n--- Final Report Summary ---")
    print(f"Topic: {result.get('topic')}")
    print(f"Papers Analyzed: {len(result.get('analyses', []))}")
    print(f"\n--- Comparison Table ---\n{result.get('comparison')}")
    print(f"\n--- Literature Review ---\n{result.get('literature_review')}")

