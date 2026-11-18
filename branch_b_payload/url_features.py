"""
url_features.py

URL feature extraction for QR Scam Detection.
Uses Tranco Top-1M as a positive reputation signal.
Exact known brand domains are not treated as typosquatting.
"""

import os
import csv
from urllib.parse import urlparse
from functools import lru_cache
from ipaddress import ip_address

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

try:
    import tldextract
    _extractor = tldextract.TLDExtract(suffix_list_urls=())
except ImportError:
    _extractor = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
TRANCO_FILE = os.path.join(DATA_DIR, "top-1m.csv")

SUSPICIOUS_KEYWORDS = [
    "verify", "login", "secure", "account", "update", "confirm",
    "otp", "bank", "refund", "prize", "urgent", "suspended", "claim",
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "ow.ly"
}

KNOWN_BRAND_DOMAINS = [
    "google.com", "amazon.com", "amazon.in", "flipkart.com",
    "hdfcbank.com", "icicibank.com", "sbi.co.in", "paytm.com",
    "irctc.co.in", "gmail.com", "instagram.com", "apple.com",
    "bookmyshow.com" ,"paypal.com",
]


def get_registered_domain(host):
    host = str(host).lower().strip(".")

    if not host:
        return ""

    if _extractor:
        result = _extractor(host)

        if result.domain and result.suffix:
            return result.top_domain_under_public_suffix

    parts = host.split(".")

    if len(parts) >= 3 and parts[-2] in {
        "co", "com", "net", "org", "gov", "ac"
    }:
        return ".".join(parts[-3:])

    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _edit_distance(a, b):
    """Fallback edit distance if python-Levenshtein is unavailable."""

    if Levenshtein is not None:
        return Levenshtein.distance(a, b)

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1

            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost
            ))

        previous = current

    return previous[-1]


@lru_cache(maxsize=1)
def load_tranco():
    """
    Loads rank,domain pairs from data/top-1m.csv.
    Tranco CSV normally has no header.
    """

    domains = {}

    if not os.path.exists(TRANCO_FILE):
        print("WARNING: Tranco file not found:", TRANCO_FILE)
        return domains

    try:
        with open(
            TRANCO_FILE, "r", encoding="utf-8", newline=""
        ) as file:

            reader = csv.reader(file)

            for row in reader:
                if len(row) < 2:
                    continue

                try:
                    rank = int(row[0].strip())
                    domain = row[1].strip().lower().rstrip(".")

                    if domain:
                        domains[domain] = rank

                except (ValueError, IndexError):
                    continue

    except OSError as error:
        print("WARNING: Could not read Tranco file:", error)

    print(f"Loaded {len(domains)} Tranco domains.")
    return domains


def get_tranco_rank(host):
    registered_domain = get_registered_domain(host)
    return load_tranco().get(registered_domain)


def _is_ip(host):
    try:
        ip_address(host)
        return True
    except ValueError:
        return False


def _min_typosquat_distance(host):
    """
    Exact known brand domains are not typosquatting.
    Only compare against other brand domains.
    """

    if not host:
        return None

    domain = get_registered_domain(host)

    if not domain:
        return None

    brand_domains = {
        get_registered_domain(brand)
        for brand in KNOWN_BRAND_DOMAINS
    }

    # Important: exact brand match must not trigger a warning.
    if domain in brand_domains:
        return 99

    distances = [
        _edit_distance(domain, brand)
        for brand in brand_domains
        if brand and brand != domain
    ]

    return min(distances) if distances else 99


def _domain_age_days(host):
    """
    Domain age is unknown because WHOIS lookups are disabled.
    -1 means unavailable.
    """
    return None


def extract_url_features(url):

    url = str(url).strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    normalized_url = (
        url if "://" in url else "http://" + url
    )

    parsed = urlparse(normalized_url)

    try:
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        host = ""

    registered_domain = (
        get_registered_domain(host) if host else ""
    )

    path_and_query = (
        (parsed.path or "") + "?" + (parsed.query or "")
    )

    subdomain_count = max(
        host.count(".") - registered_domain.count("."),
        0
    )

    typo_dist = (
        _min_typosquat_distance(host) if host else None
    )

    age_days = _domain_age_days(host) if host else None

    tranco_rank = (
        get_tranco_rank(host) if host else None
    )

    features = {
        "url_length": len(url),

        "host_length": len(host),

        "subdomain_count": subdomain_count,

        "is_ip_host": int(_is_ip(host)) if host else 0,

        "is_https": int(parsed.scheme.lower() == "https"),

        "is_shortener": int(
            registered_domain in SHORTENER_DOMAINS
        ),

        "suspicious_keyword_count": sum(
            keyword in path_and_query.lower()
            for keyword in SUSPICIOUS_KEYWORDS
        ),

        "min_typosquat_distance": (
            typo_dist if typo_dist is not None else -1
        ),

        "domain_age_days": (
            age_days if age_days is not None else -1
        ),

        "has_at_symbol": int("@" in url),

        "digit_ratio_in_host": (
            sum(char.isdigit() for char in host)
            / max(len(host), 1)
        ),

        "tranco_rank": (
            tranco_rank if tranco_rank is not None else 1000001
        ),

        "is_tranco_top_10k": int(
            tranco_rank is not None and tranco_rank <= 10000
        ),

        "is_tranco_top_50k": int(
            tranco_rank is not None and tranco_rank <= 50000
        ),
    }

    return features


def url_reasons(features):

    reasons = []

    if features["is_ip_host"]:
        reasons.append(
            "URL uses a raw IP address instead of a domain name"
        )

    if features["is_shortener"]:
        reasons.append(
            "URL uses a link-shortening service"
        )

    if features["suspicious_keyword_count"] >= 2:
        reasons.append(
            "URL path contains multiple phishing-associated keywords"
        )

    # Exact known brands have distance 99.
    # Only near matches with distance 1 or 2 are flagged.
    if 1 <= features["min_typosquat_distance"] <= 2:
        reasons.append(
            "Domain spelling is close to a known brand domain"
        )

    if 0 <= features["domain_age_days"] < 30:
        reasons.append(
            "Domain was registered very recently"
        )

    if not features["is_https"]:
        reasons.append(
            "Connection is not HTTPS"
        )

    if features["has_at_symbol"]:
        reasons.append(
            "URL contains '@', which may indicate redirect obfuscation"
        )

    # Tranco rank is a positive reputation signal.
    # Never add it as a suspicious reason.
    # Not being listed is also not proof of maliciousness.

    return reasons