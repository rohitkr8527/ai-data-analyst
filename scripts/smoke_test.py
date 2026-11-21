"""Live smoke test script for ClassicModels MySQL and Multi-Agent workflow."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

from app.services.db import ReadOnlyGateway
from app.orchestration.workflow import AnalyticalWorkflow

def main():
    print("Testing MySQL ClassicModels ReadOnlyGateway...")
    gateway = ReadOnlyGateway()
    snapshot = gateway.catalog.get_snapshot()
    print(f"Discovered tables: {list(snapshot.tables.keys())}")
    print(f"Data range: {snapshot.date_range}")

    print("\nInitializing live AnalyticalWorkflow...")
    workflow = AnalyticalWorkflow(gateway=gateway)

    test_questions = [
        "Show the top 5 products by revenue in 2004",
        "Why did sales decline in November 2004 compared with October 2004?"
    ]

    for q in test_questions:
        print(f"\nRunning question: '{q}'...")
        res = workflow.run(q)
        rep = res["report"]
        print(f"  Route: {res['route']} ({res.get('route_confidence', 1.0):.2f})")
        print(f"  Status: {rep['status']}")
        print(f"  Summary: {rep['summary']}")
        print(f"  Evidence queries: {len(rep.get('evidence', []))}")
        print(f"  Charts generated: {len(rep.get('charts', []))}")
        print(f"  Models used: {[m['model'] for m in rep.get('models_used', [])]}")

    print("\nLive smoke test completed successfully!")

if __name__ == '__main__':
    main()
