"""
fusion.py

Combines Branch A (CNN image risk) and Branch B (payload risk)
into a single explainable Low/Medium/High verdict.

Image risk can raise risk but cannot lower payload risk.
Visual anomaly alone cannot produce High Risk.
Confirmed blacklist match forces High Risk.

Official brand QR false-positive protection is applied only when
the payload analysis identifies a recognized official brand domain
and the payload score is low.
"""

import math


IMAGE_WEIGHT = 0.4
PAYLOAD_WEIGHT = 0.6

HARD_OVERRIDE_BLACKLIST_SCORE = 0.95

LOW_UPPER = 0.3
MEDIUM_UPPER = 0.7


# --------------------------------------------------
# SCORE VALIDATION
# --------------------------------------------------

def validate_score(value, name):

    try:
        score = float(value)

    except (TypeError, ValueError):

        raise ValueError(
            f"{name} must be a valid number."
        )

    if not math.isfinite(score):

        raise ValueError(
            f"{name} must be finite."
        )

    if not 0.0 <= score <= 1.0:

        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return score


# --------------------------------------------------
# RISK BAND
# --------------------------------------------------

def band(score: float) -> str:

    if score < LOW_UPPER:

        return "Low"

    elif score < MEDIUM_UPPER:

        return "Medium"

    return "High"


# --------------------------------------------------
# BLACKLIST STATUS
# --------------------------------------------------

def blacklist_status(payload_result: dict) -> str:

    if payload_result.get("blacklisted") is True:

        return "listed"

    checked = payload_result.get(
        "blacklist_checked"
    )

    if checked is True:

        return "not_found"

    if checked is False:

        return "unavailable"

    return "not_applicable"


# --------------------------------------------------
# FUSION
# --------------------------------------------------

def fuse(
    image_risk_score,
    payload_result: dict
) -> dict:

    if not isinstance(
        payload_result,
        dict
    ):

        raise TypeError(
            "payload_result must be a dictionary."
        )


    if "risk_score" not in payload_result:

        raise KeyError(
            "payload_result is missing 'risk_score'."
        )


    # ----------------------------------------------
    # PAYLOAD SCORE
    # ----------------------------------------------

    payload_score = validate_score(
        payload_result["risk_score"],
        "Payload risk score"
    )


    # ----------------------------------------------
    # IMAGE ANALYSIS UNAVAILABLE
    # ----------------------------------------------

    if image_risk_score is None:

        image_score = None

        base_score = payload_score


    # ----------------------------------------------
    # IMAGE ANALYSIS AVAILABLE
    # ----------------------------------------------

    else:

        image_score = validate_score(
            image_risk_score,
            "Image risk score"
        )


        # ------------------------------------------
        # OFFICIAL BRAND QR PROTECTION
        # ------------------------------------------

        trusted_brand_qr = (

            payload_result.get(
                "brand_trust"
            )
            in (
                "clean_root",
                "brand_path"
            )

            and payload_score <= 0.20

            and payload_result.get(
                "blacklisted"
            ) is not True

        )


        # ------------------------------------------
        # EFFECTIVE IMAGE SCORE
        # ------------------------------------------

        effective_image_score = image_score


        if trusted_brand_qr:

            effective_image_score = min(
                image_score,
                0.25
            )


        # ------------------------------------------
        # WEIGHTED SCORE
        # ------------------------------------------

        weighted_score = (

            IMAGE_WEIGHT
            * effective_image_score

            +

            PAYLOAD_WEIGHT
            * payload_score

        )


        # Image cannot reduce payload risk

        base_score = max(
            payload_score,
            weighted_score
        )


    # ----------------------------------------------
    # FINAL SCORE
    # ----------------------------------------------

    final_score = base_score

    override_triggered = False


    # ----------------------------------------------
    # BLACKLIST OVERRIDE
    # ----------------------------------------------

    if payload_result.get(
        "blacklisted"
    ) is True:

        final_score = max(
            final_score,
            HARD_OVERRIDE_BLACKLIST_SCORE
        )

        override_triggered = True


    # ----------------------------------------------
    # IMAGE-ONLY HIGH RISK PROTECTION
    # ----------------------------------------------

    elif payload_score < MEDIUM_UPPER:

        final_score = min(
            final_score,
            MEDIUM_UPPER - 0.001
        )


    # ----------------------------------------------
    # REASONS
    # ----------------------------------------------

    reasons = list(
        payload_result.get(
            "reasons"
        ) or []
    )


    # ----------------------------------------------
    # IMAGE UNAVAILABLE
    # ----------------------------------------------

    if image_score is None:

        reasons.append(
            "Image analysis was unavailable; "
            "this verdict is based on payload analysis only."
        )


    # ----------------------------------------------
    # IMAGE ANOMALY
    # ----------------------------------------------

    elif image_score >= 0.5:

        reasons.append(
            f"Visual QR anomaly detected "
            f"(CNN score: {image_score:.2f})"
        )


    # ----------------------------------------------
    # TRUSTED BRAND INFORMATION
    # ----------------------------------------------

    trusted_brand_qr = (

        payload_result.get(
            "brand_trust"
        )
        in (
            "clean_root",
            "brand_path"
        )

        and payload_score <= 0.20

        and payload_result.get(
            "blacklisted"
        ) is not True

    )


    if trusted_brand_qr:

        reasons.append(
            "Recognized official brand domain detected. "
            "The image anomaly score was limited to prevent "
            "false positives on genuine branded QR designs."
        )


    # ----------------------------------------------
    # BLACKLIST OVERRIDE REASON
    # ----------------------------------------------

    if override_triggered:

        reasons.append(
            "High Risk: URL matched a verified blacklist entry."
        )


    # ----------------------------------------------
    # IMAGE-ONLY RISK CAP
    # ----------------------------------------------

    elif (

        payload_score < MEDIUM_UPPER

        and base_score >= MEDIUM_UPPER

    ):

        reasons.append(
            "Visual anomaly alone was capped at Medium Risk."
        )


    # ----------------------------------------------
    # REMOVE DUPLICATES
    # ----------------------------------------------

    reasons = list(
        dict.fromkeys(
            reasons
        )
    )


    # ----------------------------------------------
    # FINAL OUTPUT
    # ----------------------------------------------

    return {

        "final_score": round(
            final_score,
            3
        ),

        "risk_band": band(
            final_score
        ),

        "image_available": (
            image_score is not None
        ),

        "image_risk_score": (

            round(
                image_score,
                3
            )

            if image_score is not None

            else None

        ),

        "payload_risk_score": round(
            payload_score,
            3
        ),

        "payload_type": payload_result.get(
            "payload_type"
        ),

        "reasons": reasons,

        "blacklist_override": override_triggered,

        "blacklist_checked": payload_result.get(
            "blacklist_checked"
        ),

        "blacklist_status": blacklist_status(
            payload_result
        ),

    }