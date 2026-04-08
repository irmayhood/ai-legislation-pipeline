import pandas as pd
import anthropic
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import json
import time

# Load API key from .env file
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Terms dictionary to help Claude classify correctly
TERMS_DICTIONARY = """
REGULATORY THEMES:
- Transparency/Labeling: Requirements for AI systems or outputs to be identifiable as AI-generated
- Risk/Impact Assessment: Mandates to evaluate potential harms or risks of an AI system
- Anti-bias/Civil Rights: Provisions ensuring AI systems do not produce discriminatory outcomes
- Consumer Protection (UDAP): Use of unfair/deceptive acts frameworks to regulate harmful AI
- Privacy/Data: Regulation of data collection, storage, sharing in AI systems
- Biometrics/Deepfakes: Controls over facial recognition, voiceprints, or synthetic media
- Public-Sector Governance: Frameworks for AI procurement and oversight by government
- Elections Integrity: Measures addressing AI-generated election misinformation
- Child Safety: Safeguards for minors interacting with AI
- IP/Creators: Addresses intellectual property rights for creators whose works train AI
- Safety/Cyber: Requirements ensuring system security and resilience

APPLICABILITY SCOPE (pick exactly one):
Cross-sector • Government-use • Employment/HR • Credit/Finance • Health • Education • Critical Infrastructure • Elections/Political • Advertising/Platforms • Other

AI STACK LAYER (multi-select, tag earliest touched):
Data • Training/Development • Deployment/Use • Outputs/Synthetic Media • Post-market Monitoring

REGULATED PARTIES (multi-select):
Developers • Deployers/Operators • Service Providers/Vendors • Public Agencies • Downstream Users

OBLIGATION TYPE (multi-select):
Disclosure/Labeling • Notice to Users • Impact Assessment/Audit • Risk-Management Program/Governance • Human Oversight/Contestability • Recordkeeping/Reporting • Safety Guardrails
"""

EXTRACTION_PROMPT = """Read the following bill text carefully and extract the fields below.
Return ONLY a valid JSON object with exactly these keys, nothing else before or after.
If a field cannot be determined from the bill text, use "Not specified".

Use this classification dictionary to ensure consistent coding:
{terms}

JSON keys to return:
{{
    "theme_primary": "single most relevant theme from the list",
    "theme_secondary": "second theme or Not specified",
    "applicability_scope_primary": "exactly one sector from the list",
    "applicability_scope_secondary": "second sector or Not specified",
    "ai_stack_layer_primary": "primary lifecycle stage",
    "ai_stack_layer_secondary": "second lifecycle stage or Not specified",
    "regulated_parties_primary": "primary regulated party",
    "regulated_parties_secondary": "second regulated party or Not specified",
    "obligation_type_primary": "primary obligation",
    "obligation_type_secondary": "second obligation or Not specified",
    "enforcement_regulator": "agency responsible for enforcement or Not specified",
    "enforcement_private_right_of_action": "Yes or No or Not specified",
    "enforcement_penalty_type": "civil or criminal or both or Not specified",
    "enforcement_cure_period": "Yes or No or Not specified",
    "motivation": "1-2 sentence summary of legislative intent",
    "definition_of_ai": "quote or paraphrase the bill's definition of AI, or note external source",
    "responsible_agency": "state agency or enforcement body or Not specified",
    "penalties": "dollar amount if specified or Not specified",
    "quotable_summary": "2-3 sentence academic-style summary of the bill's purpose"
}}

BILL TEXT:
{bill_text}
"""

def fetch_bill_text(url, bill_id=""):
    """Fetch and extract plain text from a bill URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)

        char_count = len(text)

        if char_count > 50000:
            print(f"  Bill {bill_id} is very long ({char_count} chars) - likely appropriations bill, using summary instead")
            return None
        elif char_count > 15000:
            print(f"  Bill {bill_id} is long ({char_count} chars) - trimming to 15000")
            return text[:15000]
        else:
            return text

    except Exception as e:
        print(f"  Failed to fetch URL: {e}")
        return None

def extract_with_claude(bill_text):
    """Send bill text to Claude and get structured JSON back"""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT.format(
                    terms=TERMS_DICTIONARY,
                    bill_text=bill_text
                )
            }]
        )
        raw = response.content[0].text.strip()
        # Remove markdown code fences if Claude added them
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  JSON parsing failed: {e}")
        print(f"  Raw response: {raw}")
        return None
    except Exception as e:
        print(f"  Claude API call failed: {e}")
        return None

def process_bills(input_csv, output_csv, limit=None):
    """Main function to process bills through Claude"""
    df = pd.read_csv(input_csv)

    if limit:
        df = df.head(limit)
        print(f"Processing {limit} bills for testing")
    else:
        print(f"Processing all {len(df)} bills")

    results = []

    for i, row in df.iterrows():
        print(f"Processing bill {i+1}/{len(df)}: {row.get('bill_id', 'unknown')}")

        bill_text = None
        if pd.notna(row.get("bill_url")):
            bill_text = fetch_bill_text(row["bill_url"], row.get("bill_id", ""))

        if not bill_text:
            print(f"  Falling back to scraped summary")
            bill_text = row.get("Summary", "No text available")

        extracted = extract_with_claude(bill_text)

        if extracted:
            combined = row.to_dict()
            combined.update(extracted)
            results.append(combined)
            print(f"  Success")
        else:
            results.append(row.to_dict())
            print(f"  Extraction failed, keeping original row")

        time.sleep(0.5)

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"\nDone! Saved {len(results)} bills to {output_csv}")

# Run on first 25 bills for testing
process_bills(
    input_csv="data/processed/ncsl_2025_scraped.csv",
    output_csv="data/processed/ncsl_2025_test.csv",
    limit=25
)