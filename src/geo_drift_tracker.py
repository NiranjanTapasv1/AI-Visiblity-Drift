import os
import time
import json
import statistics
import re
from collections import defaultdict
import argparse
from html import unescape
from urllib.error import URLError, HTTPError
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

# Configuration: list of prompts to test (editable)
PROMPTS = [
    "best CRM tools for startups",
    "top project management software",
    "best email marketing platforms"
]

# Runs per provider (keep small to respect free tiers)
N_RUNS = int(os.getenv("N_RUNS", "5"))
# Seconds to sleep between model calls
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "1.0"))

# API keys (read from .env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
if GEMINI_MODEL == "gemini-1.5-flash":
    GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = os.getenv(
    "GROQ_MODEL", "llama-3.3-70b-versatile"
)
if GROQ_MODEL == "groq-code-5b":
    GROQ_MODEL = "llama-3.3-70b-versatile"
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "5"))
WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "12"))

GENERIC_BRAND_BLACKLIST = {
    "best",
    "best crm",
    "best crm tools",
    "best email marketing platforms",
    "candidate",
    "crm",
    "crm.org",
    "crmorg",
    "discover",
    "duckduckgo",
    "growth",
    "guide",
    "platform",
    "platforms",
    "product",
    "products",
    "review",
    "reviews",
    "software",
    "startup",
    "startups",
    "startup crm tools",
    "tools",
    "top",
    "forbes advisor",
    "forrester",
    "bing",
    "capterra",
}

SYSTEM_INSTRUCTION = (
    "You are extracting brands and products from live web search context. "
    "Use only brands and products supported by the provided sources. "
    "Return 6 to 10 distinct product or brand names when possible. "
    "Do not stop after three items if more are available. "
    "For each item include a short phrase that indicates sentiment "
    "(positive|neutral|negative). Format exactly as: 1. BrandName - short description (positive|neutral|negative)\n"
)

# Optional: try to import provider SDKs; code will fall back to HTTP or to mock responses
try:
    import google.generativeai as genai
    genai_available = True
except Exception:
    genai_available = False

try:
    import groq
    groq_available = True
except Exception:
    groq_available = False


def _fetch_url(url, timeout=WEB_SEARCH_TIMEOUT):
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def _clean_html_text(text):
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_result_url(url):
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return url


def _extract_brand_candidates(text):
    candidates = set()
    patterns = [
        r"\b([A-Z][A-Za-z0-9&]+(?:\.[A-Za-z0-9&]+)?(?:\s+[A-Z][A-Za-z0-9&]+(?:\.[A-Za-z0-9&]+)?)*)\b",
        r"\b([A-Z][A-Za-z0-9&]*\.[A-Za-z0-9&]+)\b",
    ]
    stopwords = {
        "Best",
        "Top",
        "The",
        "A",
        "An",
        "And",
        "For",
        "How",
        "Why",
        "What",
        "Guide",
        "Review",
        "Reviews",
        "Software",
        "CRM",
        "Tools",
        "Platform",
        "Platforms",
        "Startups",
        "Product",
        "Products",
        "Marketing",
        "Email",
        "Management",
        "Best",
        "Top",
        "Accelerate",
        "Growth",
        "Improve",
        "Scale",
        "Increase",
        "Boost",
        "Drive",
        "Optimize",
    }
    for pattern in patterns:
        for match in re.findall(pattern, text):
            cleaned = match.strip(" ,.;:()[]{}\"'")
            if _is_plausible_brand(cleaned, stopwords):
                candidates.add(cleaned)
    return sorted(candidates)


def _is_plausible_brand(candidate, stopwords=None):
    stopwords = stopwords or set()
    cleaned = candidate.strip()
    if not cleaned:
        return False
    normalized_phrase = re.sub(r"\s+", " ", cleaned).lower()
    if normalized_phrase in GENERIC_BRAND_BLACKLIST:
        return False
    if len(cleaned) < 2:
        return False
    tokens = cleaned.split()
    lower_tokens = [token.lower() for token in tokens]
    if len(tokens) > 3:
        return False
    if cleaned.lower().startswith(("best ", "top ", "guide ", "review ", "free ", "cheap ")):
        return False
    if any(token in GENERIC_BRAND_BLACKLIST for token in lower_tokens):
        if len(tokens) == 1 or all(token in GENERIC_BRAND_BLACKLIST for token in lower_tokens):
            return False
    if any(token.lower() in stopwords for token in tokens):
        if all(token.lower() in stopwords for token in tokens):
            return False
    if "&cid" in cleaned.lower():
        return False
    alnum = re.sub(r"[^A-Za-z0-9]", "", cleaned)
    if re.fullmatch(r"[A-F0-9]{10,}", alnum.upper()):
        return False
    if re.search(r"[A-F0-9]{16,}", alnum.upper()):
        return False
    if re.search(r"^[^A-Za-z]*$", cleaned):
        return False
    if len(alnum) >= 12 and sum(ch.isdigit() for ch in alnum) >= 4:
        return False
    if cleaned.upper() == cleaned and len(cleaned) <= 4:
        return False
    if len(tokens) == 1 and lower_tokens[0] in stopwords:
        return False
    if all(token.lower() in stopwords for token in tokens):
        return False
    if len(tokens) > 1 and all(token[0].isupper() and token[1:].islower() for token in tokens):
        # Reject generic title-case phrases like "Accelerate Growth" unless
        # they include a brand-like token with mixed case or digits.
        if not any(re.search(r"[a-z][A-Z]", token) or any(ch.isdigit() for ch in token) for token in tokens):
            return False
    if len(tokens) > 4:
        return False
    return True


def retrieve_live_sources(prompt, max_results=WEB_SEARCH_RESULTS):
    """Fetch live search snippets that ground the model responses."""
    if not WEB_SEARCH_ENABLED:
        return "", []

    query = quote_plus(prompt)
    search_url = f"https://html.duckduckgo.com/html/?q={query}"
    try:
        html_text = _fetch_url(search_url)
        title_matches = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html_text,
            re.S,
        )
        snippet_matches = re.findall(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html_text,
            re.S,
        )
        results = []
        all_brand_candidates = set()
        source_text_for_candidates = []

        for idx, (href, title_html) in enumerate(title_matches[:max_results]):
            title = _clean_html_text(title_html)
            snippet = ""
            if idx < len(snippet_matches):
                snippet = _clean_html_text(snippet_matches[idx])
            url = _normalize_result_url(href)
            results.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                }
            )
            source_text_for_candidates.append(f"{title} {snippet}")

        all_brand_candidates.update(
            _extract_brand_candidates(" ".join(source_text_for_candidates))
        )

        if not results:
            return "", []

        source_lines = []
        for i, item in enumerate(results, 1):
            source_lines.append(
                f"[{i}] {item['title']}\nSnippet: {item['snippet']}\nURL: {item['url']}"
            )

        if all_brand_candidates:
            source_lines.append(
                "Candidate brands mentioned across sources: "
                + ", ".join(list(all_brand_candidates)[:20])
            )

        return "\n\n".join(source_lines), results
    except (URLError, HTTPError, TimeoutError, ValueError) as e:
        print(f"Web retrieval failed: {e}")
        return "", []
    except Exception as e:
        print(f"Web retrieval failed: {e}")
        return "", []


def _build_grounded_prompt(prompt, web_context=None):
    if not web_context:
        return prompt
    return (
        f"{prompt}\n\n"
        "Live web sources:\n"
        f"{web_context}\n\n"
        "Use the live sources as grounding. Return only brands or products that appear in the sources. "
        "Prefer six to ten distinct items when enough are available."
    )


def extract_brands_from_sources(prompt, web_context):
    """Use a grounded model pass to extract real brand names from live sources."""
    if not web_context:
        return []

    extractor_instruction = (
        "Extract only real software, product, or company brand names that are explicitly mentioned "
        "in the live sources. Do not return generic words, adjectives, article titles, URLs, or phrases. "
        "Return 6 to 10 names when available. Respond as a plain comma-separated list and nothing else."
    )

    extractor_prompt = (
        f"User query: {prompt}\n\n"
        f"Live sources:\n{web_context}\n\n"
        "Return only the brand names from the live sources."
    )

    if GEMINI_API_KEY and genai_available:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=extractor_instruction,
            )
            resp = model.generate_content(extractor_prompt)
            raw = (resp.text or "").strip()
            if not raw:
                return []
            parts = re.split(r"[\n,;•]+", raw)
            brands = []
            seen = set()
            for part in parts:
                cleaned = part.strip().strip("-*0123456789. ")
                cleaned = re.sub(r"\s+", " ", cleaned)
                if not cleaned:
                    continue
                if cleaned.lower() in seen:
                    continue
                if _is_plausible_brand(cleaned):
                    brands.append(cleaned)
                    seen.add(cleaned.lower())
            return _filter_brands_present_in_context(brands, web_context)
        except Exception as e:
            print(f"Brand extraction failed: {e}")

    return []


def _filter_brands_present_in_context(candidate_brands, web_context):
    if not candidate_brands or not web_context:
        return candidate_brands or []

    normalized_context = re.sub(r"[^A-Za-z0-9]+", " ", web_context).lower()
    filtered = []
    seen = set()
    for brand in candidate_brands:
        normalized_brand = re.sub(r"[^A-Za-z0-9]+", "", brand).lower()
        if not normalized_brand:
            continue
        if normalized_brand in normalized_context and normalized_brand not in seen:
            filtered.append(brand)
            seen.add(normalized_brand)
    return filtered


def _supplement_with_live_candidates(parsed_results, candidate_brands, web_context, minimum=6):
    """Add live-source brand candidates when the model returns too few items."""
    candidate_brands = _filter_brands_present_in_context(candidate_brands, web_context)
    if len(parsed_results) >= minimum or not candidate_brands:
        return parsed_results

    seen = {name.strip().lower() for name, _, _ in parsed_results}
    supplemented = list(parsed_results)
    next_position = max([pos for _, pos, _ in parsed_results if isinstance(pos, int)], default=0) + 1

    for candidate in candidate_brands:
        cleaned = candidate.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        supplemented.append((cleaned, next_position, "neutral"))
        seen.add(key)
        next_position += 1
        if len(supplemented) >= minimum:
            break

    return supplemented


def query_gemini(prompt, run_index=0):
    system_instruction = (
        "You are a helpful assistant. "
        "When asked about software tools or products, "
        "respond with a numbered list of exactly "
        "6 to 8 items. For each item write the "
        "product name first, then a dash, then one "
        "sentence description. "
        "Example format:\n"
        "1. HubSpot - A popular CRM for growing teams.\n"
        "2. Salesforce - Enterprise CRM with advanced features.\n"
        "Be direct. No disclaimers. No extra text."
    )
    if GEMINI_API_KEY and genai_available:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=system_instruction,
            )
            response = model.generate_content(prompt)
            time.sleep(SLEEP_BETWEEN_CALLS)
            print(f"  [Gemini REAL] run {run_index}: "
                  f"{prompt[:35]}...")
            return response.text
        except Exception as e:
            print(f"  [Gemini FAILED] {e}")
    print(f"  [Gemini MOCK] using fallback data")
    mock_responses = [
        "1. HubSpot - Great CRM for startups.\n"
        "2. Salesforce - Powerful enterprise CRM.\n"
        "3. Pipedrive - Simple pipeline management.\n"
        "4. Zoho CRM - Affordable and flexible.\n"
        "5. Freshsales - Easy to use sales CRM.\n"
        "6. Monday.com - Visual project management.",

        "1. Salesforce - Leading enterprise CRM.\n"
        "2. HubSpot - Popular inbound marketing CRM.\n"
        "3. Zoho CRM - Budget friendly option.\n"
        "4. Pipedrive - Great for sales teams.\n"
        "5. Notion - Flexible workspace tool.\n"
        "6. Asana - Strong project tracking.",

        "1. HubSpot - Best for inbound teams.\n"
        "2. Monday.com - Visual and intuitive.\n"
        "3. Salesforce - Enterprise standard.\n"
        "4. Freshsales - Good for small teams.\n"
        "5. Pipedrive - Pipeline focused CRM.\n"
        "6. Zoho CRM - Comprehensive suite.",

        "1. Pipedrive - Top visual pipeline tool.\n"
        "2. HubSpot - Marketing and sales combined.\n"
        "3. Salesforce - Most powerful CRM.\n"
        "4. Asana - Great task management.\n"
        "5. Notion - Popular team workspace.\n"
        "6. Monday.com - Easy to adopt.",

        "1. HubSpot - Most popular startup CRM.\n"
        "2. Salesforce - Enterprise grade platform.\n"
        "3. Pipedrive - Clean pipeline view.\n"
        "4. Zoho CRM - Value for money.\n"
        "5. Freshsales - Modern sales CRM.\n"
        "6. Asana - Reliable project tool.",
    ]
    return mock_responses[run_index % len(mock_responses)]


def query_groq(prompt, run_index=0):
    system_instruction = (
        "You are a helpful assistant. "
        "When asked about software tools or products, "
        "respond with a numbered list of exactly "
        "6 to 8 items. For each item write the "
        "product name first, then a dash, then one "
        "sentence description. "
        "Example format:\n"
        "1. HubSpot - A popular CRM for growing teams.\n"
        "2. Salesforce - Enterprise CRM with advanced features.\n"
        "Be direct. No disclaimers. No extra text."
    )
    if GROQ_API_KEY and groq_available:
        try:
            client = groq.Groq(api_key=GROQ_API_KEY)
            chat = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system",
                     "content": system_instruction},
                    {"role": "user",
                     "content": prompt},
                ],
                temperature=0.9,
                max_tokens=600,
            )
            time.sleep(SLEEP_BETWEEN_CALLS)
            print(f"  [Groq REAL] run {run_index}: "
                  f"{prompt[:35]}...")
            return chat.choices[0].message.content
        except Exception as e:
            print(f"  [Groq FAILED] {e}")
    print(f"  [Groq MOCK] using fallback data")
    mock_responses = [
        "1. Salesforce - Industry leading CRM.\n"
        "2. HubSpot - Great for growing companies.\n"
        "3. Monday.com - Visual project tool.\n"
        "4. Pipedrive - Sales pipeline focused.\n"
        "5. Asana - Reliable task manager.\n"
        "6. Zoho CRM - Affordable option.",

        "1. HubSpot - Inbound marketing leader.\n"
        "2. Salesforce - Enterprise powerhouse.\n"
        "3. Asana - Clean project management.\n"
        "4. Pipedrive - Simple pipeline tool.\n"
        "5. Notion - Flexible team workspace.\n"
        "6. Monday.com - Easy visual planning.",

        "1. Pipedrive - Best pipeline visual.\n"
        "2. Zoho CRM - Budget CRM option.\n"
        "3. HubSpot - Marketing and sales hub.\n"
        "4. Salesforce - Most feature rich.\n"
        "5. Monday.com - Good for teams.\n"
        "6. Freshsales - Modern CRM choice.",

        "1. HubSpot - Popular with startups.\n"
        "2. Asana - Strong for task tracking.\n"
        "3. Salesforce - Enterprise standard.\n"
        "4. Zoho CRM - Great value.\n"
        "5. Pipedrive - Sales team favorite.\n"
        "6. Notion - Collaborative workspace.",

        "1. Salesforce - Top enterprise CRM.\n"
        "2. HubSpot - Best for small teams.\n"
        "3. Monday.com - Visual management.\n"
        "4. Asana - Task tracking leader.\n"
        "5. Pipedrive - Clean sales view.\n"
        "6. Zoho CRM - Affordable suite.",
    ]
    return mock_responses[run_index % len(mock_responses)]


def parse_response(text):
    """Parse the model text into a list of (brand, position, sentiment).

    Assumptions:
    - The model returns numbered lines like: `1. BrandName - description (positive)`
    - If sentiment isn't explicit, we guess from positive/negative words in the description.
    - Brands are taken as the phrase before the first `-`.
    """
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*(\d+)[\)\.\-:\s]+(.+?)\s*-\s*(.+)\((positive|neutral|negative)\)\s*$", line, re.I)
        if m:
            pos = int(m.group(1))
            name = m.group(2).strip()
            desc = m.group(3).strip()
            sentiment = m.group(4).lower()
            if _is_plausible_brand(name):
                results.append((name, pos, sentiment))
            continue

        # fallback: try to parse "1. Brand - description"
        m2 = re.match(r"^\s*(\d+)[\)\.\-:\s]+([^\-]+)\s*-\s*(.+)$", line)
        if m2:
            pos = int(m2.group(1))
            name = m2.group(2).strip()
            desc = m2.group(3).strip()
            # simple sentiment guess
            if re.search(r"good|great|best|excellent|positive|love", desc, re.I):
                sentiment = "positive"
            elif re.search(r"bad|poor|terrible|hate|negative", desc, re.I):
                sentiment = "negative"
            else:
                sentiment = "neutral"
            if _is_plausible_brand(name):
                results.append((name, pos, sentiment))
            continue

        # final fallback: find capitalized words sequences as brand names
        names = re.findall(r"\b([A-Z][a-zA-Z0-9&]+(?:\s+[A-Z][a-zA-Z0-9&]+)*)\b", line)
        if names:
            name = names[0]
            if _is_plausible_brand(name):
                results.append((name, 999, "neutral"))

    return results


def aggregate_runs(runs, n_runs):
    """Aggregate multiple runs for a single provider.

    runs: list of lists of tuples (name, position, sentiment)
    Returns a DataFrame with computed metrics per brand.
    """
    brands = defaultdict(lambda: {"positions": [], "sentiments": []})
    for run in runs:
        seen_in_run = set()
        for name, pos, sentiment in run:
            brands[name]["positions"].append(pos)
            brands[name]["sentiments"].append(sentiment)
            seen_in_run.add(name)
        # mark absent brands implicitly by not adding positions for that run

    rows = []
    for name, data in brands.items():
        positions = data["positions"]
        freq = len(positions)
        avg_pos = statistics.mean(positions) if positions else None
        std_pos = statistics.pstdev(positions) if positions and len(positions) > 1 else 0.0
        # Stability score heuristic (0..1): combines frequency and position variance
        presence_score = freq / n_runs
        stability_from_variance = 1 / (1 + std_pos)
        stability = presence_score * stability_from_variance
        # most common sentiment
        sentiment = None
        if data["sentiments"]:
            sentiment = max(set(data["sentiments"]), key=data["sentiments"].count)

        rows.append({
            "brand": name,
            "mention_frequency": freq,
            "mention_rate": freq / n_runs,
            "avg_position": avg_pos,
            "position_std": std_pos,
            "stability": round(stability, 3),
            "dominant_sentiment": sentiment,
        })

    df = pd.DataFrame(rows).sort_values(["stability", "mention_frequency"], ascending=[False, False])
    return df


def run_pipeline(prompts, n_runs=N_RUNS, providers=('gemini', 'groq')):
    all_provider_results = {p: {} for p in providers}
    raw_outputs = {p: {} for p in providers}

    for prompt in prompts:
        print(f"\n=== Prompt: {prompt}\n")
        web_context, live_sources = retrieve_live_sources(prompt)
        candidate_brands = extract_brands_from_sources(prompt, web_context)
        if not candidate_brands:
            candidate_brands = _extract_brand_candidates(web_context)
        candidate_brands = _filter_brands_present_in_context(candidate_brands, web_context)
        if web_context:
            print(f"Retrieved {len(live_sources)} live sources for grounding.")
            if candidate_brands:
                print("Live candidate brands:", ", ".join(candidate_brands[:12]))
        else:
            print("No live sources retrieved; falling back to model-only answers.")
        gemini_runs = []
        groq_runs = []
        gemini_texts = []
        groq_texts = []
        for i in range(n_runs):
            if 'gemini' in providers:
                print(f"Gemini run {i+1}/{n_runs}...", end=" ")
                g_text = query_gemini(
                    _build_grounded_prompt(
                        prompt,
                        web_context
                        + (
                            "\n\nLive candidate brands: " + ", ".join(candidate_brands[:20])
                            if candidate_brands
                            else ""
                        ),
                    ),
                    run_index=i,
                )
                print("done")
                gemini_texts.append(g_text)
                parsed_g = parse_response(g_text)
                gemini_runs.append(_supplement_with_live_candidates(parsed_g, candidate_brands, web_context))
                time.sleep(SLEEP_BETWEEN_CALLS)

            if 'groq' in providers:
                print(f"Groq run {i+1}/{n_runs}...", end=" ")
                q_text = query_groq(
                    _build_grounded_prompt(
                        prompt,
                        web_context
                        + (
                            "\n\nLive candidate brands: " + ", ".join(candidate_brands[:20])
                            if candidate_brands
                            else ""
                        ),
                    ),
                    run_index=i,
                )
                print("done")
                groq_texts.append(q_text)
                parsed_q = parse_response(q_text)
                groq_runs.append(_supplement_with_live_candidates(parsed_q, candidate_brands, web_context))
                time.sleep(SLEEP_BETWEEN_CALLS)

        safe_prompt = re.sub(r"[^0-9A-Za-z]+", "_", prompt)[:50]

        if 'gemini' in providers:
            gemini_df = aggregate_runs(gemini_runs, n_runs)
            all_provider_results['gemini'][prompt] = gemini_df
            raw_outputs['gemini'][prompt] = {'texts': gemini_texts, 'parsed': gemini_runs}
            gemini_df.to_csv(f"results/gemini_{safe_prompt}.csv", index=False)
            print("\nGemini summary:")
            print(gemini_df.to_string(index=False))

        if 'groq' in providers:
            groq_df = aggregate_runs(groq_runs, n_runs)
            all_provider_results['groq'][prompt] = groq_df
            raw_outputs['groq'][prompt] = {'texts': groq_texts, 'parsed': groq_runs}
            groq_df.to_csv(f"results/groq_{safe_prompt}.csv", index=False)
            print("\nGroq summary:")
            print(groq_df.to_string(index=False))

        # Visualization (only if both providers present)
        if 'gemini' in providers and 'groq' in providers:
            visualize_comparison(gemini_df, groq_df, prompt)
        else:
            print('Skipping comparison chart (need both providers).')

    return all_provider_results, raw_outputs


def combine_and_report(all_provider_results, raw_outputs, prompts):
    """Create combined CSVs and a short textual summary per prompt.

    Returns a dict mapping prompt -> {'summary': str, 'combined_csv': path}.
    """
    summaries = {}
    os.makedirs('results', exist_ok=True)
    for prompt in prompts:
        safe_prompt = re.sub(r"[^0-9A-Za-z]+", "_", prompt)[:50]
        # handle missing providers gracefully
        a = all_provider_results.get('gemini', {}).get(prompt, pd.DataFrame(columns=['brand']))
        b = all_provider_results.get('groq', {}).get(prompt, pd.DataFrame(columns=['brand']))

        if a.empty and not b.empty:
            a = pd.DataFrame(columns=['brand'])
        if b.empty and not a.empty:
            b = pd.DataFrame(columns=['brand'])

        a_k = a.set_index('brand') if not a.empty else pd.DataFrame()
        b_k = b.set_index('brand') if not b.empty else pd.DataFrame()

        brands = sorted(set(a['brand']).union(set(b['brand'])))
        rows = []
        for brand in brands:
            row = {'brand': brand}
            if not a.empty and brand in list(a['brand']):
                ra = a_k.loc[brand]
                row.update({
                    'gemini_mention_rate': ra['mention_rate'],
                    'gemini_avg_position': ra['avg_position'],
                    'gemini_position_std': ra['position_std'],
                    'gemini_stability': ra['stability'],
                })
            else:
                row.update({
                    'gemini_mention_rate': 0,
                    'gemini_avg_position': None,
                    'gemini_position_std': None,
                    'gemini_stability': 0,
                })
            if not b.empty and brand in list(b['brand']):
                rb = b_k.loc[brand]
                row.update({
                    'groq_mention_rate': rb['mention_rate'],
                    'groq_avg_position': rb['avg_position'],
                    'groq_position_std': rb['position_std'],
                    'groq_stability': rb['stability'],
                })
            else:
                row.update({
                    'groq_mention_rate': 0,
                    'groq_avg_position': None,
                    'groq_position_std': None,
                    'groq_stability': 0,
                })
            # disagreement metric
            try:
                if row['gemini_avg_position'] is not None and row['groq_avg_position'] is not None:
                    row['avg_position_diff'] = abs(row['gemini_avg_position'] - row['groq_avg_position'])
                else:
                    row['avg_position_diff'] = None
            except Exception:
                row['avg_position_diff'] = None

            rows.append(row)

        combined_df = pd.DataFrame(rows).sort_values(['gemini_stability', 'groq_stability'], ascending=False)
        combined_path = f"results/combined_{safe_prompt}.csv"
        combined_df.to_csv(combined_path, index=False)

        # generate a short textual summary
        combined_df['mean_stability'] = combined_df[['gemini_stability', 'groq_stability']].mean(axis=1)
        stable = combined_df.sort_values('mean_stability', ascending=False).head(3)
        volatile = combined_df.sort_values('mean_stability', ascending=True).head(3)

        summary_lines = [f"Summary for prompt: {prompt}", ""]
        summary_lines.append("Top stable brands:")
        for _, r in stable.iterrows():
            summary_lines.append(f"- {r['brand']}: mean_stability={r['mean_stability']:.3f}, gemini_avg={r['gemini_avg_position']}, groq_avg={r['groq_avg_position']}")
        summary_lines.append("")
        summary_lines.append("Top volatile brands:")
        for _, r in volatile.iterrows():
            summary_lines.append(f"- {r['brand']}: mean_stability={r['mean_stability']:.3f}, gemini_std={r.get('gemini_position_std')}, groq_std={r.get('groq_position_std')}")
        summary_lines.append("")
        disagree = combined_df.dropna(subset=['avg_position_diff']).sort_values('avg_position_diff', ascending=False).head(3)
        summary_lines.append("Top disagreements (by avg position diff):")
        for _, r in disagree.iterrows():
            summary_lines.append(f"- {r['brand']}: position diff={r['avg_position_diff']}")

        summary = "\n".join(summary_lines)
        with open(f"results/summary_{safe_prompt}.txt", 'w') as f:
            f.write(summary)

        summaries[prompt] = {'summary': summary, 'combined_csv': combined_path}

    return summaries


def generate_markdown_report(combined_info, prompts, outpath='results/report.md'):
    """Generate a polished markdown report summarizing insights and embedding charts."""
    lines = ["# GEO Drift Tracker — Report", ""]
    for prompt in prompts:
        safe_prompt = re.sub(r"[^0-9A-Za-z]+", "_", prompt)[:50]
        info = combined_info.get(prompt, {})
        summary_text = info.get('summary', '').splitlines()
        # pick a short plain-language insight: the first non-empty line after the header
        insight = ''
        for ln in summary_text:
            if ln.strip() and not ln.startswith('Top') and not ln.startswith('Summary'):
                insight = ln
                break
        if not insight:
            insight = 'No clear insight generated.'

        lines.append(f"## Prompt: {prompt}")
        lines.append("")
        lines.append("**Single most useful insight:**")
        lines.append("")
        lines.append(insight)
        lines.append("")

        combined_csv = info.get('combined_csv')
        if combined_csv and os.path.exists(combined_csv):
            df = pd.read_csv(combined_csv)
            lines.append("**Combined provider comparison:**")
            lines.append("")
            lines.append(df.to_markdown(index=False))
            lines.append("")

        chart_path = f"results/compare_{safe_prompt}.png"
        if os.path.exists(chart_path):
            lines.append(f"![comparison chart]({chart_path})")
            lines.append("")

    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Wrote Markdown report to {outpath}")


def visualize_comparison(df_a, df_b, prompt):
    """Plot average positions with error bars for two providers side-by-side."""
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6))

    # Prepare combined set of brands
    brands = sorted(set(df_a['brand']).union(set(df_b['brand'])))
    x = range(len(brands))

    a_means = [df_a.set_index('brand').loc[b]['avg_position'] if b in list(df_a['brand']) else None for b in brands]
    a_errs = [df_a.set_index('brand').loc[b]['position_std'] if b in list(df_a['brand']) else 0 for b in brands]
    b_means = [df_b.set_index('brand').loc[b]['avg_position'] if b in list(df_b['brand']) else None for b in brands]
    b_errs = [df_b.set_index('brand').loc[b]['position_std'] if b in list(df_b['brand']) else 0 for b in brands]

    # scatter lines (handle None by replacing with max+1)
    maxpos = max([m for m in (a_means + b_means) if m is not None] + [len(brands)])
    a_plot = [m if m is not None else maxpos + 1 for m in a_means]
    b_plot = [m if m is not None else maxpos + 1 for m in b_means]

    plt.errorbar([i - 0.1 for i in x], a_plot, yerr=a_errs, fmt='o', label='Gemini')
    plt.errorbar([i + 0.1 for i in x], b_plot, yerr=b_errs, fmt='s', label='Groq')

    plt.xticks(x, brands, rotation=45, ha='right')
    plt.gca().invert_yaxis()
    plt.ylabel('Average Position (lower is better)')
    plt.title(f'Provider comparison — {prompt}')
    plt.legend()
    plt.tight_layout()
    outpath = f"results/compare_{re.sub(r'[^0-9A-Za-z]+','_',prompt)[:50]}.png"
    plt.savefig(outpath)
    plt.close()
    print(f"Saved comparison chart to {outpath}")


def test_api_connections():
    print("\n=== TESTING API CONNECTIONS ===")
    print(f"GEMINI_API_KEY present: "
          f"{'YES' if GEMINI_API_KEY else 'NO'}")
    print(f"GROQ_API_KEY present: "
          f"{'YES' if GROQ_API_KEY else 'NO'}")
    print(f"genai available: {genai_available}")
    print(f"groq available: {groq_available}")
    print(f"GROQ_MODEL: {GROQ_MODEL}")
    print("\nTesting Gemini...")
    result = query_gemini("best CRM tools", run_index=0)
    print(f"Gemini result preview: {result[:80]}")
    print("\nTesting Groq...")
    result = query_groq("best CRM tools", run_index=0)
    print(f"Groq result preview: {result[:80]}")
    print("\n=== TEST COMPLETE ===\n")


if __name__ == "__main__":
    test_api_connections()
