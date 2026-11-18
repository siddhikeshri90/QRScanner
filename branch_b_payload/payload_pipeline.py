"""
payload_pipeline.py

Branch B entry point:
Routes decoded QR payloads and calculates explainable risk scores.

Important:
- ML probability is not treated as guaranteed real-world risk.
- Popular/known domains are supporting signals, never proof of safety.
- Known-brand domains do not blindly suppress ML risk.
- A confirmed blacklist match remains a strong warning.
- Blacklist lookup failure is not treated as proof of safety.
- URL shorteners and embedded redirects receive additional risk.
- Recognized official brand URLs receive false-positive protection
  only when multiple safety conditions are satisfied.
"""

import os
import re
import sys

from urllib.parse import (
    urlparse,
    parse_qs,
    unquote
)

import joblib
import pandas as pd


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# --------------------------------------------------
# IMPORT BRANCH B MODULES
# --------------------------------------------------

from payload_type import classify_payload

from url_features import (
    extract_url_features,
    url_reasons,
    get_registered_domain,
    KNOWN_BRAND_DOMAINS,
)

from upi_analysis import analyze_upi

from text_analysis import (
    analyze_text,
    analyze_vcard,
    analyze_wifi
)

from blacklist_check import check_phishtank


# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "url_model.joblib"
)

FEATURE_ORDER_PATH = os.path.join(
    os.path.dirname(__file__),
    "url_feature_order.joblib"
)


# --------------------------------------------------
# MODEL CACHE
# --------------------------------------------------

_url_model = None

_feature_order = None


# --------------------------------------------------
# RISK SCORING CONSTANTS
# --------------------------------------------------

REDIRECT_FLOOR = 0.60

SHORTENER_FLOOR = 0.45

OFFICIAL_BRAND_ML_CAP = 0.15


# --------------------------------------------------
# OFFICIAL BRAND DOMAINS
# --------------------------------------------------

OFFICIAL_BRAND_DOMAINS = {

    domain.lower()

    for domain in KNOWN_BRAND_DOMAINS

}


# --------------------------------------------------
# REDIRECT PARAMETERS
# --------------------------------------------------

REDIRECT_PARAM_NAMES = {

    "url",
    "u",
    "q",
    "redirect",
    "redirect_uri",
    "redirect_url",
    "next",
    "continue",
    "dest",
    "destination",
    "target",
    "return",
    "returnurl",
    "return_to",
    "goto",
    "link",
    "out",
    "r",

}


# --------------------------------------------------
# URL REGEX
# --------------------------------------------------

_EMBEDDED_URL_RE = re.compile(
    r"(?i)(?:https?:)?//[a-z0-9.-]+\.[a-z]{2,}"
)


_DOMAIN_LIKE_RE = re.compile(
    r"(?i)^[a-z0-9.-]+\.[a-z]{2,}(?:[/:?#].*)?$"
)


# --------------------------------------------------
# LOAD URL MODEL
# --------------------------------------------------

def _load_url_model():

    global _url_model

    global _feature_order


    if _url_model is None:

        if not os.path.exists(
            MODEL_PATH
        ):

            return None, None


        if not os.path.exists(
            FEATURE_ORDER_PATH
        ):

            raise FileNotFoundError(
                "URL feature-order file is missing."
            )


        _url_model = joblib.load(
            MODEL_PATH
        )


        _feature_order = joblib.load(
            FEATURE_ORDER_PATH
        )


    return (
        _url_model,
        _feature_order
    )


# --------------------------------------------------
# ML PREDICTION
# --------------------------------------------------

def _predict_ml_probability(
    model,
    feature_order,
    features
):

    """
    Predict with exact saved feature order.

    Never reorders features.

    Sends a named DataFrame if the model was
    fitted with feature names.

    Otherwise sends a NumPy array in the exact
    saved feature order.
    """

    feature_order = list(
        feature_order
    )


    missing = [

        feature

        for feature in feature_order

        if feature not in features

    ]


    if missing:

        raise ValueError(
            f"Missing URL model features: {missing}"
        )


    expected = getattr(
        model,
        "feature_names_in_",
        None
    )


    if expected is not None:

        if list(expected) != feature_order:

            raise ValueError(
                "url_model.joblib and "
                "url_feature_order.joblib do not match. "
                "Retrain using train_url_classifier.py."
            )


    row = pd.DataFrame(
        [
            [

                features[feature]

                for feature in feature_order

            ]
        ],
        columns=feature_order
    )


    # Model trained with feature names.

    if expected is not None:

        probabilities = model.predict_proba(
            row
        )[0]


    # Model trained without feature names.

    else:

        probabilities = model.predict_proba(
            row.to_numpy()
        )[0]


    classes = list(
        model.classes_
    )


    if 1 not in classes:

        raise ValueError(
            "Malicious class label 1 "
            "was not found in the URL model."
        )


    malicious_index = classes.index(
        1
    )


    probability = probabilities[
        malicious_index
    ]


    return float(
        probability
    )


# --------------------------------------------------
# URL PARSING
# --------------------------------------------------

def _parse_url(url):

    """
    Returns parsed URL.

    Returns None if malformed.
    """

    try:

        parsed = urlparse(
            url
            if "://" in url
            else "http://" + url
        )


        # Validate hostname.

        _ = parsed.hostname


        # Validate port.

        _ = parsed.port


        return parsed


    except ValueError:

        return None


# --------------------------------------------------
# DOUBLE URL DECODING
# --------------------------------------------------

def _decode_twice(text):

    return unquote(
        unquote(text)
    )


# --------------------------------------------------
# REDIRECT DETECTION
# --------------------------------------------------

def _has_redirect_target(parsed):

    """
    Detects embedded redirect destinations
    in path, query or fragment.
    """

    if parsed is None:

        return False


    tail = _decode_twice(
        parsed.path
        + "?"
        + parsed.query
        + "#"
        + parsed.fragment
    )


    # Embedded URL detection.

    if _EMBEDDED_URL_RE.search(
        tail
    ):

        return True


    # Redirect parameter detection.

    query_parameters = parse_qs(
        parsed.query
    )


    for name, values in query_parameters.items():

        if name.lower() in REDIRECT_PARAM_NAMES:

            for value in values:

                decoded_value = _decode_twice(
                    value
                ).strip()


                if _DOMAIN_LIKE_RE.match(
                    decoded_value
                ):

                    return True


    return False


# --------------------------------------------------
# BRAND DOMAIN INFORMATION
# --------------------------------------------------

def _brand_trust_level(
    url,
    features
):

    """
    Returns:

        None
        clean_root
        brand_path
        redirect

    Brand recognition is informational only.

    It does not automatically reduce risk.
    """

    parsed = _parse_url(
        url
    )


    if parsed is None:

        return None


    host = (

        parsed.hostname

        or ""

    ).lower().rstrip(".")


    registered_domain = get_registered_domain(
        host
    )


    # Not a recognized official brand.

    if registered_domain not in OFFICIAL_BRAND_DOMAINS:

        return None


    # HTTPS required for brand recognition.

    if parsed.scheme.lower() != "https":

        return None


    # Non-standard ports are not trusted.

    if parsed.port not in (
        None,
        443
    ):

        return None


    # Reject internationalized/punycode domains.

    if "xn--" in host:

        return None


    # Reject suspicious host characteristics.

    if features.get(
        "is_ip_host",
        False
    ):

        return None


    if features.get(
        "has_at_symbol",
        False
    ):

        return None


    if features.get(
        "is_shortener",
        False
    ):

        return None


    if features.get(
        "min_typosquat_distance"
    ) in (
        1,
        2
    ):

        return None


    # Embedded redirect.

    if _has_redirect_target(
        parsed
    ):

        return "redirect"


    plain_host = host in (
        registered_domain,
        "www." + registered_domain
    )


    is_root = (

        parsed.path in (
            "",
            "/"
        )

        and not parsed.query

        and not parsed.fragment

    )


    if plain_host and is_root:

        return "clean_root"


    return "brand_path"


# --------------------------------------------------
# URL PAYLOAD ANALYSIS
# --------------------------------------------------

def analyze_url_payload(
    url: str,
    phishtank_api_key: str = None
) -> dict:


    # ----------------------------------------------
    # FEATURE EXTRACTION
    # ----------------------------------------------

    features = extract_url_features(
        url
    )


    # ----------------------------------------------
    # INITIAL REASONS
    # ----------------------------------------------

    reasons = url_reasons(
        features
    )


    # ----------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------

    model, feature_order = _load_url_model()


    # ----------------------------------------------
    # ML PROBABILITY
    # ----------------------------------------------

    if model is not None:

        ml_probability = _predict_ml_probability(
            model,
            feature_order,
            features
        )


    else:

        # Fallback heuristic.

        ml_probability = min(
            0.15
            * features[
                "suspicious_keyword_count"
            ],
            1.0
        )


        reasons.append(
            "Trained URL model unavailable; "
            "using fallback heuristic."
        )


    # ----------------------------------------------
    # PHISHTANK BLACKLIST
    # ----------------------------------------------

    blacklist_result = check_phishtank(
        url,
        api_key=phishtank_api_key
    )


    blacklisted = (
        blacklist_result.get(
            "in_blacklist"
        ) is True
    )


    blacklist_checked = (
        blacklist_result.get(
            "checked"
        ) is True
    )


    # ----------------------------------------------
    # BRAND INFORMATION
    # ----------------------------------------------

    trust = _brand_trust_level(
        url,
        features
    )


    # ----------------------------------------------
    # OFFICIAL BRAND FALSE-POSITIVE PROTECTION
    # ----------------------------------------------

    parsed_url = _parse_url(
        url
    )


    official_brand = (

        trust in (
            "clean_root",
            "brand_path"
        )

        and features.get(
            "is_https",
            False
        )

        and not features.get(
            "is_shortener",
            False
        )

        and not features.get(
            "is_ip_host",
            False
        )

        and not features.get(
            "has_at_symbol",
            False
        )

        and features.get(
            "min_typosquat_distance",
            99
        ) not in (
            1,
            2
        )

        and not _has_redirect_target(
            parsed_url
        )

        and not blacklisted

    )


    if official_brand:

        # Reduce ML false positives on
        # recognized official brand domains.

        ml_probability = min(
            ml_probability,
            OFFICIAL_BRAND_ML_CAP
        )


        reasons.append(
            "Recognized official brand domain detected. "
            "ML probability was adjusted to reduce "
            "potential false-positive classification."
        )


    # ----------------------------------------------
    # INITIAL RISK SCORE
    # ----------------------------------------------

    final_score = ml_probability


    # ----------------------------------------------
    # CONFIRMED BLACKLIST
    # ----------------------------------------------

    if blacklisted:

        final_score = 0.99


        reasons.append(
            "URL matched an entry in the "
            "PhishTank phishing blacklist."
        )


    else:


        # ------------------------------------------
        # BLACKLIST UNAVAILABLE
        # ------------------------------------------

        if not blacklist_checked:

            reasons.append(
                "Blacklist verification unavailable; "
                "URL safety is unconfirmed."
            )


        # ------------------------------------------
        # BRAND INFORMATION
        # ------------------------------------------

        if trust == "clean_root":

            reasons.append(
                "URL belongs to a recognized official "
                "brand root domain over HTTPS. "
                "Brand recognition is not proof of safety."
            )


        elif trust == "brand_path":

            reasons.append(
                "URL uses a recognized brand domain "
                "with an additional path or query. "
                "The domain's popularity does not "
                "guarantee the destination is safe."
            )


        elif trust == "redirect":

            final_score = max(
                final_score,
                REDIRECT_FLOOR
            )


            reasons.append(
                "URL contains an embedded redirect target. "
                "The final destination may differ from "
                "the visible URL."
            )


        # ------------------------------------------
        # URL SHORTENER
        # ------------------------------------------

        if features.get(
            "is_shortener",
            False
        ):

            final_score = max(
                final_score,
                SHORTENER_FLOOR
            )


            reasons.append(
                "Shortened link hides its real destination. "
                "The destination is unverified."
            )


        # ------------------------------------------
        # HIGH ML RISK
        # ------------------------------------------

        if ml_probability >= 0.70:

            reasons.append(
                "URL classifier assigned a high "
                "phishing probability. "
                "Further verification is recommended."
            )


    # ----------------------------------------------
    # FINAL RESULT
    # ----------------------------------------------

    return {

        "risk_score": round(
            final_score,
            4
        ),

        "reasons": list(
            dict.fromkeys(
                reasons
            )
        ),

        "ml_probability": round(
            ml_probability,
            4
        ),

        "blacklisted": blacklisted,

        "blacklist_checked": blacklist_checked,

        "brand_trust": trust,

    }


# --------------------------------------------------
# GENERIC PAYLOAD ANALYSIS
# --------------------------------------------------

def analyze_payload(
    payload: str,
    phishtank_api_key: str = None
) -> dict:


    payload = str(
        payload
    ).strip()


    # ----------------------------------------------
    # CLASSIFY PAYLOAD
    # ----------------------------------------------

    payload_type = classify_payload(
        payload
    )


    # ----------------------------------------------
    # ROUTE TO ANALYZER
    # ----------------------------------------------

    if payload_type == "url":

        result = analyze_url_payload(
            payload,
            phishtank_api_key
        )


    elif payload_type == "upi":

        result = analyze_upi(
            payload
        )


    elif payload_type == "vcard":

        result = analyze_vcard(
            payload
        )


    elif payload_type == "wifi":

        result = analyze_wifi(
            payload
        )


    else:

        result = analyze_text(
            payload
        )


    # ----------------------------------------------
    # ADD PAYLOAD TYPE
    # ----------------------------------------------

    result["payload_type"] = payload_type


    return result