import sys
sys.path.insert(0, "branch_b_payload")
from upi_analysis import analyze_upi
from payload_pipeline import analyze_payload

tests = [
    ("mobile@yespop, no amount",  "upi://pay?pa=9876543210@yespop&pn=Ravi%20Kumar&cu=INR"),
    ("unknown handle",            "upi://pay?pa=shop@newpsp&pn=Cafe"),
    ("scam wording",              "upi://pay?pa=refund.kyc@ybl&pn=Refund%20Dept&am=499&cu=INR"),
    ("duplicate pa",              "upi://pay?pa=ab@ybl&pa=cd@ybl&pn=Shop"),
    ("missing pa",                "upi://pay?pn=Shop&am=100"),
    ("currency USD",              "upi://pay?pa=shop@ybl&pn=Shop&cu=USD"),
    ("embedded url",              "upi://pay?pa=shop@ybl&pn=Shop&url=http://evil.com"),
    ("malformed",                 "upi://[pay?pa=x"),
    ("no amount, clean",          "upi://pay?pa=shop@ybl&pn=Shop"),
]

for name, p in tests:
    r = analyze_upi(p)
    print(f"\n{name}: score={r['risk_score']} handle={r['handle_status']} status={r['analysis_status']}")
    for x in r["reasons"]:
        print("   -", x)

print("\nPipeline routing:", analyze_payload(tests[0][1])["payload_type"])