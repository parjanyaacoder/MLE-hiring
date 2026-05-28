import re
import base64

# Regex patterns for PII detection
PII_PATTERNS = {
    "credit_card": re.compile(
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b'
        r'|\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        r'|\b\d{4}[-\s]?[xX\*]{4}[-\s]?[xX\*]{4}[-\s]?\d{4}\b'
    ),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "email": re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
    "phone": re.compile(
        r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        # Chinese/international variants like +86 138 XXXX 7890 or 138-xxxx-7890
        r'|\b(?:\+86\s?)?1[3-9]\d[-\s\*\?]{1,9}\d{4}\b'
    ),
    # Common zip codes and addresses (e.g., 1847 Oak Street, Springfield, IL 62704)
    "zip_code": re.compile(r'\b\d{5}(?:-\d{4})?\b'),
    "address": re.compile(
        r'\b\d+\s+[A-Za-z0-9\.\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Circle|Cir)\b',
        re.IGNORECASE
    )
}

# Key terms used in jailbreak attempts or system prompt extraction
INJECTION_KEYWORDS = [
    "ignore all previous instructions",
    "disregard all previous instructions",
    "ignore the guidelines",
    "ignore system",
    "override safety",
    "system override",
    "dan mode enabled",
    "dan (do anything now)",
    "output the following exactly",
    "you are no longer a support",
    "act as my personal financial advisor",
    "disclose your system instructions",
    "reveal your prompt",
    "what is your system prompt",
    "system maintenance in progress",
    "alert_ack_cve-",
    "ignore all guidelines",
    "ignore instructions"
]

def scan_pii(text):
    """
    Checks if any PII is present in the text.
    Returns a dictionary of found PII types and Boolean indicating if any was found.
    """
    if not text:
        return False, {}
        
    found_pii = {}
    any_detected = False
    
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found_pii[pii_type] = matches
            any_detected = True
            
    return any_detected, found_pii

def scrub_pii(text):
    """
    Redacts specific PII instances in a text, replacing them with generic equivalents.
    Useful as a post-processing step on LLM responses to ensure no PII is accidentally echoed.
    """
    if not text:
        return text

    scrubbed = text

    # Redact credit card numbers
    scrubbed = PII_PATTERNS["credit_card"].sub("[CREDIT CARD]", scrubbed)

    # Redact SSNs
    scrubbed = PII_PATTERNS["ssn"].sub("[SSN]", scrubbed)

    # Redact emails
    scrubbed = PII_PATTERNS["email"].sub("[EMAIL]", scrubbed)

    # Redact phones
    scrubbed = PII_PATTERNS["phone"].sub("[PHONE NUMBER]", scrubbed)

    return scrubbed

def check_base64_injection(text):
    """
    Checks if a string contains base64-encoded text that translates to an injection attempt.
    """
    if not text:
        return False, ""
        
    # Look for base64 blocks: standard alphanumeric + '+' + '/' + '=' (at least 12 chars to avoid false positives)
    b64_candidates = re.findall(r'\b[A-Za-z0-9+/]{12,}=*\b', text)
    
    for candidate in b64_candidates:
        try:
            decoded = base64.b64decode(candidate).decode('utf-8', errors='ignore').lower()
            # Check if decoded content contains injection keywords
            for keyword in INJECTION_KEYWORDS:
                if keyword in decoded:
                    return True, f"Base64-encoded injection detected: '{keyword}'"
        except Exception:
            continue
            
    return False, ""

def is_excel_formula(text):
    """
    Detects if the query is an Excel formula injection attempt, which begins with = or contains cmd|'
    """
    if not text:
        return False
    trimmed = text.strip()
    # Excel formula characters: =, +, -, @
    if trimmed.startswith("=") or trimmed.startswith("+") or trimmed.startswith("-") or trimmed.startswith("@"):
        if "cmd" in trimmed.lower() or "calc" in trimmed.lower() or "|" in trimmed:
            return True
    return False

def scan_injection(text):
    """
    Performs multiple checks to determine if the input contains a prompt injection/jailbreak attempt.
    Returns (is_injection, reason_string)
    """
    if not text:
        return False, ""

    text_lower = text.lower()

    # 1. Direct Keyword Check
    for keyword in INJECTION_KEYWORDS:
        if keyword in text_lower:
            return True, f"Direct injection keyword matching: '{keyword}'"

    # 2. Base64 Check
    is_b64, b64_reason = check_base64_injection(text)
    if is_b64:
        return True, b64_reason

    # 3. Excel Formula Check
    if is_excel_formula(text):
        return True, "Excel Formula injection attempt detected"

    # 4. XML / System tag overrides
    if "<system>" in text_lower or "[system override]" in text_lower or "[system]" in text_lower:
        return True, "System tag override attempt detected"

    return False, ""
