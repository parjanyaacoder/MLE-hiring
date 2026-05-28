import os
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from config import DATA_DIR, REPO_ROOT

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cant", "cannot",
    "co", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each",
    "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed",
    "hell", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id",
    "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", "itself", "lets", "me", "more", "most",
    "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should",
    "shouldnt", "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves",
    "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "werent",
    "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys",
    "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
    "yourselves"
}

def tokenize(text):
    if not text:
        return []
    text = text.lower()
    # Replace non-alphanumeric (except dashes and dots in URLs/emails/errors) with spaces
    text = re.sub(r'[^a-z0-9\-\.\_\@\/\:]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def extract_date_from_content(content, filepath=""):
    """
    Looks for date patterns (e.g., Q1 2026, 2026, 2025, 2024, Updated/Created dates) in the content.
    Returns a score indicating how recent the document is (larger is more recent).
    """
    # Look for years in the file name or content
    years = [int(y) for y in re.findall(r'\b(202\d)\b', content + " " + filepath)]
    if years:
        max_year = max(years)
        # Check for quarter details
        quarters = re.findall(r'\b[qQ]([1-4])\b', content)
        max_q = 0
        if quarters:
            max_q = max(int(q) for q in quarters)
        return max_year * 10 + max_q
    
    # Default score for neutral/unknown dates
    return 2020 * 10

class DocumentRetriever:
    def __init__(self):
        self.docs = []
        self.bm25 = None
        self.build_index()

    def build_index(self):
        print("Crawling and indexing support corpus...")
        if not DATA_DIR.exists():
            print(f"Warning: Corpus directory {DATA_DIR} not found.")
            return

        for root, _, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith(".md"):
                    full_path = Path(root) / file
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        
                        relative_path = os.path.relpath(full_path, REPO_ROOT)
                        # Specificity: path depth and not being a general index.md
                        depth = len(Path(relative_path).parts)
                        is_index = file.lower() == "index.md"
                        
                        # Recency score
                        recency_score = extract_date_from_content(content, str(relative_path))
                        
                        self.docs.append({
                            "path": relative_path,
                            "filename": file,
                            "content": content,
                            "depth": depth,
                            "is_index": is_index,
                            "recency_score": recency_score,
                            "tokens": tokenize(content + " " + file)
                        })
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")

        if self.docs:
            tokenized_corpus = [doc["tokens"] for doc in self.docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"Indexed {len(self.docs)} documents successfully.")
        else:
            print("No documents found to index.")

    def search(self, query, company=None, top_k=5):
        """
        Searches the indexed corpus using BM25 and applies heuristics for company boosting,
        specificity (preferring deeper paths over index.md), and recency.
        """
        if not self.bm25 or not query:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Get initial BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        scored_docs = []
        for idx, doc in enumerate(self.docs):
            score = scores[idx]
            if score <= 0:
                continue
            
            # 1. Company Boosting
            # If the document belongs to the target company's directory, boost the score
            if company:
                company_lower = company.lower()
                # Directory structure: data/claude/..., data/visa/..., data/devplatform/...
                path_parts = [part.lower() for part in Path(doc["path"]).parts]
                if len(path_parts) > 1 and path_parts[1] == company_lower:
                    score *= 1.5
                elif company_lower == "devplatform" and "devplatform" in path_parts:
                    score *= 1.5
            
            # 2. Specificity Adjustment
            # Deeper paths are more specific. Penalize index.md overview files.
            if doc["is_index"]:
                score *= 0.5
            else:
                # Slight boost for deeper paths
                score *= (1.0 + (doc["depth"] - 2) * 0.05)
                
            # 3. Recency Boosting
            # Slight boost for more recent documents
            # E.g., year 2026 vs 2024
            year_diff = (doc["recency_score"] - 2020 * 10) / 10.0
            if year_diff > 0:
                score *= (1.0 + min(year_diff * 0.02, 0.2)) # Max 20% boost for recent updates
                
            scored_docs.append((doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Check for contradictions/disagreements among top results
        results = [item[0] for item in scored_docs[:top_k]]
        return results

    def resolve_conflicts(self, matched_docs):
        """
        Analyzes the retrieved documents to see if there is a conflict or disagreement.
        Returns a boolean indicating if a conflict was detected.
        """
        if len(matched_docs) < 2:
            return False
            
        # Example check: look for different numeric limits or opposite instructions in the top 2 documents
        # For simplicity, we can inspect if there's key term differences like refund limits
        # We'll let the LLM handle the deep semantic resolution, but here we can flag if documents from
        # different sub-folders disagree.
        first_doc = matched_docs[0]
        for other_doc in matched_docs[1:3]:
            # If both are about the same topic but from different directories or different dates, flag potential warning
            if first_doc["filename"] == other_doc["filename"] and first_doc["path"] != other_doc["path"]:
                return True
        return False
