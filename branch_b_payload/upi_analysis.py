"""
upi_analysis.py

Rule-based analysis of upi://pay payloads.

Indicators are kept in separate tiers:
  suspicious  - can be verified from the payload itself
  wording     - heuristic scam wording
  unverified  - could not be verified locally (tiny penalty only)
  info        - normal UPI features (no penalty)

A low score is NOT a safety verdict: this module cannot verify who owns
the payee account.
"""

import re
from collections import Counter
from urllib.parse import urlparse, parse_qsl

# Local reference list. NOT the official NPCI list, so a handle missing
# from it is "unverified", never "malicious". Extend as needed.
KNOWN_PSP_HANDLES = {
    "okhdfcbank", "okaxis", "oksbi", "okicici",
    "ybl", "ibl", "axl", "paytm", "apl", "upi",
    "yapl", "ptyes", "ptaxis", "pthdfc", "ptsbi",
    "yespop", "superyes",
    "waicici", "waaxis", "wahdfcbank", "wasbi",
    "icici", "hdfcbank", "sbi",
}

# Long, unambiguous scam words: substring match.
SCAM_SUBSTRINGS = (
    "refund", "penalty", "prize", "lottery", "kyc", "challan",
    "parkingfine", "trafficfine",
)
# Short words that appear inside normal words: whole-token match only.
SCAM_TOKENS = {"fine", "claim", "reward", "winner"}

GENERIC_PAYEE_NAMES = {"merchant", "user", "unknown", "test", "na", "n/a"}

VPA_RE = re.compile(r"^[A-Za-z0-9._\-]{2,64}@[A-Za-z][A-Za-z0-9]{1,31}$")
AMOUNT_RE = re.compile(r"^\d+(\.\d{1,2})?$")
URL_RE = re.compile(r"(?i)(https?://|www\.)\S+")


def _clean(value, limit=40):
    """Keep only harmless characters so attacker text cannot inject markup."""
    return re.sub(r"[^A-Za-z0-9._@\- ]", "?", str(value))[:limit]


def _clean_field(value):
    return re.sub(r"[\x00-\x1f]", "", str(value))[:200]


def _scam_words(text):
    text = str(text).lower()
    hits = [w for w in SCAM_SUBSTRINGS if w in text]
    tokens = set(re.findall(r"[a-z]+", text))
    hits += [w for w in SCAM_TOKENS if w in tokens]
    return hits


def parse_upi(payload: str) -> dict:
    """Backward-compatible helper: first value of each parameter."""
    parsed = urlparse(payload)
    out = {}
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        out.setdefault(key.lower(), value)
    return out


def analyze_upi(payload: str) -> dict:
    suspicious, wording, unverified, info = [], [], [], []
    risk = 0.0

    try:
        parsed = urlparse(str(payload).strip())
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
    except ValueError:
        return {
            "risk_score": 0.4,
            "reasons": [
                "UPI payload is malformed and could not be parsed. "
                "Do not pay using this QR code."
            ],
            "fields": {},
            "analysis_status": "malformed",
            "handle_status": "unknown",
            "suspicious_indicators": ["malformed_payload"],
            "unverified_indicators": [],
            "info_notes": [],
            "embedded_urls": [],
        }

    # ---- parameters (first value wins; duplicates are flagged) ----
    params = {}
    for key, value in pairs:
        params.setdefault(key.lower().strip(), value.strip())

    counts = Counter(key.lower().strip() for key, _ in pairs)
    duplicated = sorted(k for k, c in counts.items() if c > 1)
    if duplicated:
        names = ", ".join(_clean(k, 10) for k in duplicated[:4])
        suspicious.append(
            f"UPI link repeats parameter(s) ({names}). Apps may use a "
            "different value than the one analysed here."
        )
        risk += 0.30

    # ---- payee address (pa) ----
    pa = params.get("pa", "")
    handle_status = "missing"

    if not pa:
        suspicious.append(
            "Payee UPI ID (pa) is missing from the payment link."
        )
        risk += 0.25
    elif pa.count("@") != 1 or not VPA_RE.match(pa):
        handle_status = "malformed"
        suspicious.append(
            "Payee UPI ID has an unusual or malformed format."
        )
        risk += 0.20
    else:
        local, handle = pa.split("@")
        handle_l = handle.lower()

        if handle_l in KNOWN_PSP_HANDLES:
            handle_status = "recognized"
        else:
            handle_status = "unrecognized"
            unverified.append(
                f"UPI handle '@{_clean(handle_l, 20)}' is not in this "
                "app's local reference list. That does not mean it is "
                "malicious; the list is incomplete."
            )
            risk += 0.05

        if local.isdigit() and len(local) == 10:
            info.append(
                "Payee ID is a mobile number. This is a normal UPI "
                "format and is not scored as suspicious."
            )

    if pa and _scam_words(pa.split("@")[0]):
        wording.append(
            "UPI ID contains scam-associated wording "
            "(refund, fine, prize, KYC, etc.)."
        )
        risk += 0.25

    # ---- payee name (pn) and note (tn) ----
    pn = params.get("pn", "")
    tn = params.get("tn", "")

    if not pn or pn.strip().lower() in GENERIC_PAYEE_NAMES:
        unverified.append(
            "Payee name is missing or generic, so the payee cannot be "
            "checked by name."
        )
        risk += 0.05

    if _scam_words(pn) or _scam_words(tn):
        wording.append(
            "Payee name or payment note contains scam-associated wording."
        )
        risk += 0.20

    # ---- currency (cu) ----
    cu = params.get("cu", "")
    if cu and cu.upper() != "INR":
        suspicious.append(
            "Payment currency is not INR, which is unusual for a UPI QR."
        )
        risk += 0.30

    # ---- amount (am): absent is fine; present is only info ----
    am = params.get("am", "")
    if am:
        if not AMOUNT_RE.match(am) or float(am) <= 0:
            suspicious.append(
                "Payment amount is invalid (not a positive number with "
                "up to two decimals)."
            )
            risk += 0.25
        else:
            info.append(
                f"Amount is pre-filled (Rs {_clean(am, 12)}). Normal for "
                "bills and dynamic merchant QR codes; check it before "
                "paying."
            )

    # ---- embedded URLs (url / tn) ----
    embedded_urls = []
    for field in ("url", "tn"):
        value = params.get(field, "")
        if value and URL_RE.search(value):
            embedded_urls.append(_clean_field(value))

    if "url" in params and params["url"]:
        embedded_urls = list(dict.fromkeys(embedded_urls))

    if embedded_urls:
        suspicious.append(
            "UPI link contains an embedded web address. It should be "
            "checked separately before opening."
        )
        risk += 0.20

    # ---- build reasons ----
    reasons = suspicious + wording + unverified + info

    if suspicious or wording:
        reasons.append(
            "Do not pay unless you can verify the payee outside this "
            "QR code."
        )
    else:
        reasons.append(
            "No suspicious indicators were found in the payload itself. "
            "This does not verify the payee: confirm the payee name "
            "shown in your UPI app before paying."
        )

    risk = round(min(risk, 1.0), 4)

    return {
        "risk_score": risk,
        "reasons": list(dict.fromkeys(reasons)),
        "fields": {
            _clean(k, 20): _clean_field(v) for k, v in params.items()
        },
        "analysis_status": "ok",
        "handle_status": handle_status,
        "suspicious_indicators": suspicious + wording,
        "unverified_indicators": unverified,
        "info_notes": info,
        "embedded_urls": embedded_urls,
    }
