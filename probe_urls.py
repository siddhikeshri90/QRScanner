import os, sys
import pandas as pd

sys.path.insert(0, "branch_b_payload")
from url_features import extract_url_features, get_registered_domain, SHORTENER_DOMAINS
from payload_pipeline import analyze_url_payload

tests = [
    "https://www.paypal.com",
    "https://www.google.com",
    "https://bit.ly/xk93jd",
    "https://bit.ly",
    "https://www.google.com/url?q=http://evil.com",
    "https://paypa1-verify.com/login",
    "https://www.google.com/search?q=cats",
"https://www.google.com/url?q=https%3A%2F%2Fevil.com",
]

for url in tests:
    f = extract_url_features(url)
    r = analyze_url_payload(url)
    print(f"\n{url}")
    print("  tranco_rank:", f["tranco_rank"], "| top10k:", f["is_tranco_top_10k"],
          "| shortener:", f["is_shortener"], "| typo_dist:", f["min_typosquat_distance"])
    print("  ml_prob:", r["ml_probability"], "| final:", r["risk_score"],
          "| blacklist_checked:", r["blacklist_checked"])

df = pd.read_csv(os.path.join("data", "urls.csv"))
df["rd"] = df["url"].astype(str).map(
    lambda u: get_registered_domain(u.split("//")[-1].split("/")[0]))
print("\nShortener rows in urls.csv by label:")
print(df[df.rd.isin(SHORTENER_DOMAINS)].groupby("label").size())
print("\npaypal rows:", (df.rd == "paypal.com").sum(),
      "| google rows:", (df.rd == "google.com").sum())