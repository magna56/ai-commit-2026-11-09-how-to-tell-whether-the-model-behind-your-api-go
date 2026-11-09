# How to Tell Whether the Model Behind Your API Got Worse

**TL;DR:** To detect an LLM API quality drop, count the responses that fail a cheap check instead of averaging an eval score. How rarely that check fires on healthy replies decides how fast you see the drop.

Published from [The AI Commit](https://theaicommit.com/#2026-11-09/code) — Evals & Reliability, 2026-11-09.

## Run

```bash
python3 code_example.py
```

## Output

```
baseline: 51 of 20000 healthy responses trip a check (0.26%)

requests per window to catch the drop (1% false alerts, 90% power)
 bug rate  average score  count, clean check  count, noisy check
     0.8%        441,898               6,488              50,458
     3.0%         31,424               1,088               4,179
    16.0%          1,105                 162                 263
  clean check = trips on 0.2% of healthy replies; noisy = 3%

replay at 3% bug rate, 1,088 requests per window:
  alerts on 50 buggy windows:   50
  alerts on 50 healthy windows: 0

```

## Code

See [`code_example.py`](code_example.py).
