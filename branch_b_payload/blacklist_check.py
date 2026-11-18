"""
blacklist_check.py

PhishTank URL verification with safe response handling.
"""

import requests

PHISHTANK_CHECK_URL = "https://checkurl.phishtank.com/checkurl/"
USER_AGENT = "phishtank/qrshield-student"


def _is_true(value):
    """Accept only explicit positive values."""
    return value is True or (
        isinstance(value, str)
        and value.strip().lower() in ("y", "true")
    )


def check_phishtank(url: str, api_key: str = None, timeout: int = 10) -> dict:
    """
    Three possible states:
    Confirmed match: checked=True, in_blacklist=True
    Not listed:      checked=True, in_blacklist=False
    Unknown:         checked=False, in_blacklist=False
    """

    data = {
        "url": url,
        "format": "json"
    }

    if api_key:
        data["app_key"] = api_key

    try:
        response = requests.post(
            PHISHTANK_CHECK_URL,
            data=data,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout
        )

        response.raise_for_status()

        if "json" not in response.headers.get("content-type", "").lower():
            return {
                "in_blacklist": False,
                "checked": False,
                "error": "PhishTank returned a non-JSON response."
            }

        body = response.json()

        if not isinstance(body, dict):
            raise ValueError("Invalid JSON response structure.")

        results = body.get("results")

        if not isinstance(results, dict):
            raise ValueError("Missing or invalid results field.")

        if "in_database" not in results or "valid" not in results:
            raise ValueError("Missing blacklist verification fields.")

        in_database = _is_true(results["in_database"])
        valid = _is_true(results["valid"])

        confirmed = in_database and valid

        return {
            "in_blacklist": confirmed,
            "checked": True,
            "error": None
        }

    except requests.RequestException as exc:
        return {
            "in_blacklist": False,
            "checked": False,
            "error": str(exc)
        }

    except (ValueError, TypeError, KeyError) as exc:
        return {
            "in_blacklist": False,
            "checked": False,
            "error": str(exc)
        }
