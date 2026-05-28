import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import INPUT_CSV, OUTPUT_CSV, EXPECTED_HEADERS, MAX_WORKERS
from retriever import DocumentRetriever
from agent import SupportAgent

def main():
    start_time = time.time()
    print("=" * 60)
    print("Starting Multi-Domain Support Triage Agent")
    print("=" * 60)
    
    # 1. Initialize retriever and index corpus
    retriever = DocumentRetriever()
    
    # 2. Initialize support agent
    agent = SupportAgent(retriever=retriever)
    
    # 3. Read tickets from input CSV
    tickets = []
    print(f"Loading input tickets from {INPUT_CSV}...")
    with open(INPUT_CSV, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            tickets.append(row)
            
    print(f"Loaded {len(tickets)} tickets.")
    
    # 4. Process tickets concurrently
    print("Processing tickets concurrently with ThreadPoolExecutor...")
    results = [None] * len(tickets)  # Pre-allocate list to maintain order
    
    # Use config MAX_WORKERS to pace requests (essential for Free Tier rate limit compliance)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks with their index to preserve ordering in the output CSV
        future_to_index = {
            executor.submit(agent.process_ticket, ticket): i 
            for i, ticket in enumerate(tickets)
        }
        
        # Track progress with tqdm
        for future in tqdm(as_completed(future_to_index), total=len(tickets), desc="Tickets Processed"):
            idx = future_to_index[future]
            try:
                processed_row = future.result()
                results[idx] = processed_row
            except Exception as e:
                print(f"\nTask index {idx} generated an exception: {e}")
                # Fallback in case of thread execution crash
                results[idx] = {
                    "issue": tickets[idx].get("Issue", ""),
                    "subject": tickets[idx].get("Subject", ""),
                    "company": tickets[idx].get("Company", ""),
                    "response": "We encountered an error processing this request and have escalated it to a human agent.",
                    "product_area": "general_support",
                    "status": "escalated",
                    "request_type": "product_issue",
                    "justification": f"Thread execution exception: {str(e)}",
                    "confidence_score": "0.10",
                    "source_documents": "",
                    "risk_level": "medium",
                    "pii_detected": "false",
                    "language": "en",
                    "actions_taken": "[]"
                }

    # 5. Write to output CSV
    print(f"Writing outputs to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=EXPECTED_HEADERS)
        writer.writeheader()
        writer.writerows(results)
        
    end_time = time.time()
    elapsed = end_time - start_time
    print("=" * 60)
    print(f"Processing Complete!")
    print(f"Total time elapsed: {elapsed:.2f} seconds")
    print(f"Average time per ticket: {elapsed / len(tickets):.2f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    main()
