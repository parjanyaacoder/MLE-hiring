import csv
from config import INPUT_CSV
from retriever import DocumentRetriever
from agent import SupportAgent

def main():
    print("Initializing test run...")
    retriever = DocumentRetriever()
    agent = SupportAgent(retriever)
    
    # Read the first ticket
    with open(INPUT_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_ticket = next(reader)
        
    print("\n--- Processing First Ticket ---")
    print(f"Subject: {first_ticket.get('Subject')}")
    print(f"Company: {first_ticket.get('Company')}")
    print(f"Issue: {first_ticket.get('Issue')[:200]}...")
    
    result = agent.process_ticket(first_ticket)
    
    print("\n--- Result ---")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
