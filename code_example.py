"""Catch a silent quality drop in a hosted model by counting broken responses.

Implements the three pieces from the article:
  1. per-response checks that a healthy model almost never fails
  2. a drift detector that compares a recent window of failures to a baseline
  3. a sample-size calculator: how many requests before a drop of a given size shows

Then it replays a serving bug like the ones in Anthropic's 2025 postmortem (a small
share of responses get stray foreign characters or broken code) and compares the
counter against the usual eval: averaging a quality score over the same requests.

Run: python3 code_example.py
Edit the four knobs below to match your own traffic and checks.
"""
import ast, json, math, random, re

BUG_RATE = 0.03        # share of responses the serving bug touches; the postmortem saw 0.8% to 16%
CATCH_RATE = 0.5       # share of touched responses your checks notice (the model often recovers)
FALSE_ALARM = 0.002    # share of HEALTHY responses that trip a check anyway; the knob that matters most
Z_ALPHA, Z_POWER = 2.33, 1.28   # 1% false alerts, 90% chance of catching a real drop

# ---- 1. per-response checks: cheap, deterministic, run on every response ------------------
UNEXPECTED_SCRIPT = re.compile(r"[\u0e00-\u0e7f\u4e00-\u9fff]")   # Thai, CJK in an English reply

def check_response(text, expect="prose"):
    """Return the names of the checks this response fails. Empty list means it looks healthy."""
    failed = []
    if UNEXPECTED_SCRIPT.search(text):
        failed.append("unexpected_script")
    if expect == "python":
        try:
            ast.parse(text)
        except SyntaxError:
            failed.append("code_does_not_parse")
    if expect == "json":
        try:
            json.loads(text)
        except ValueError:
            failed.append("json_does_not_parse")
    return failed

# ---- 2. drift detector: failures in the last window vs a baseline window ------------------
class DriftDetector:
    def __init__(self, baseline_failures, baseline_total):
        self.b_fail, self.b_total = baseline_failures, baseline_total
        self.fail = self.total = 0

    def record(self, failed_checks):
        self.total += 1
        self.fail += bool(failed_checks)

    def z(self):
        # two-proportion z-test; pooled rate so a quiet baseline does not divide by zero
        if not self.total:
            return 0.0
        pooled = (self.b_fail + self.fail) / (self.b_total + self.total)
        se = math.sqrt(max(pooled * (1 - pooled) * (1 / self.b_total + 1 / self.total), 1e-12))
        return (self.fail / self.total - self.b_fail / self.b_total) / se

    def alert(self):
        return self.z() > Z_ALPHA

# ---- 3. how many requests a window needs ----------------------------------------------------
def requests_needed_count(false_alarm, bug_rate, catch_rate):
    b0, b1 = false_alarm, false_alarm + bug_rate * catch_rate
    bar = (b0 + b1) / 2
    top = Z_ALPHA * math.sqrt(2 * bar * (1 - bar)) + Z_POWER * math.sqrt(b0 * (1 - b0) + b1 * (1 - b1))
    return math.ceil((top / (b1 - b0)) ** 2)

def requests_needed_mean(bug_rate, score=0.8, score_sd=0.25, damage=0.3):
    # a touched response loses `damage` of its score; the mean moves by bug_rate * damage * score
    shift = bug_rate * damage * score
    return math.ceil(2 * ((Z_ALPHA + Z_POWER) * score_sd / shift) ** 2)

# ---- demo: a fake model behind a serving bug -------------------------------------------------
HEALTHY = {"prose": "The cache key must include the model version.",
           "python": "def total(xs):\n    return sum(xs)\n",
           "json": '{"status": "ok", "items": 3}'}

def serve(kind, rng, bug_rate):
    text = HEALTHY[kind]
    if rng.random() < FALSE_ALARM:                 # a healthy oddity: a quoted Thai name, say
        text += " \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35"
    if rng.random() < bug_rate and rng.random() < CATCH_RATE:
        cut = rng.randrange(1, len(text))          # a high-probability wrong token mid-reply
        bad = {"prose": "ก", "python": "((", "json": '"'}[kind]   # what one stray token breaks
        text = text[:cut] + bad + text[cut:]
    return text

def run_window(n, bug_rate, rng, baseline=(0, 1)):
    det = DriftDetector(*baseline)
    for i in range(n):
        kind = ("prose", "python", "json")[i % 3]
        det.record(check_response(serve(kind, rng, bug_rate), kind))
    return det

def main():
    rng = random.Random(7)
    base = run_window(20_000, 0.0, rng)            # a week of healthy traffic sets the baseline
    BASELINE = (base.fail, base.total)
    print(f"baseline: {base.fail} of {base.total} healthy responses trip a check "
          f"({base.fail / base.total:.2%})\n")

    print("requests per window to catch the drop (1% false alerts, 90% power)")
    print(f"{'bug rate':>9} {'average score':>14} {'count, clean check':>19} {'count, noisy check':>19}")
    for p in (0.008, 0.03, 0.16):
        print(f"{p:>9.1%} {requests_needed_mean(p):>14,} "
              f"{requests_needed_count(FALSE_ALARM, p, CATCH_RATE):>19,} "
              f"{requests_needed_count(0.03, p, CATCH_RATE):>19,}")
    print("  clean check = trips on 0.2% of healthy replies; noisy = 3%\n")

    n = requests_needed_count(FALSE_ALARM, BUG_RATE, CATCH_RATE)
    hits = sum(run_window(n, BUG_RATE, rng, BASELINE).alert() for _ in range(50))
    quiet = sum(run_window(n, 0.0, rng, BASELINE).alert() for _ in range(50))
    print(f"replay at {BUG_RATE:.0%} bug rate, {n:,} requests per window:")
    print(f"  alerts on 50 buggy windows:   {hits}")
    print(f"  alerts on 50 healthy windows: {quiet}")

if __name__ == "__main__":
    main()
