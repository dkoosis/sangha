#!/usr/bin/env python3
"""Mini semantic-conditioning run. Stdlib only (urllib). Underpowered: 1 sample/cond."""
import json, os, re, urllib.request, sys

MODEL = "claude-haiku-4-5-20251001"
KEY = os.environ["ANTHROPIC_API_KEY"]

CONDITIONS = {
    "control":        "",
    "generic_hq":     "Write high-quality, production-ready code.\n\n",
    "greek_arete":    "Write code embodying ἀρετή (excellence).\n\n",
    "jp_shokunin":    "Write code with 職人気質 (craftsman spirit).\n\n",
    "combined":       "Write code embodying ἀρετή and 職人気質.\n\n",
}

# (name, prompt, test-source). Tests probe correctness incl. edge cases.
PROBLEMS = [
    ("has_close_elements",
     "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Return True if any two numbers are closer to each other than the given threshold.\"\"\"\n",
     "assert has_close_elements([1.0,2.0,3.0],0.5)==False\nassert has_close_elements([1.0,2.8,3.0,4.0,5.0,2.0],0.3)==True\nassert has_close_elements([],1.0)==False\nassert has_close_elements([1.1,2.2,3.1,4.1,5.1],1.0)==True\n"),
    ("truncate_number",
     "def truncate_number(number: float) -> float:\n    \"\"\"Return the decimal part of a positive float (number minus its integer part).\"\"\"\n",
     "assert abs(truncate_number(3.5)-0.5)<1e-6\nassert abs(truncate_number(1.0)-0.0)<1e-6\nassert abs(truncate_number(123.456)-0.456)<1e-4\n"),
    ("rolling_max",
     "from typing import List\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\"From a list of integers, return a list of the running maximum found until each position.\"\"\"\n",
     "assert rolling_max([1,2,3,2,3,4,2])==[1,2,3,3,3,4,4]\nassert rolling_max([])==[]\nassert rolling_max([4,3,2,1])==[4,4,4,4]\nassert rolling_max([3])==[3]\n"),
    ("parse_paren_depth",
     "def parse_paren_depth(s: str) -> int:\n    \"\"\"Given a string of parentheses, return the maximum nesting depth.\"\"\"\n",
     "assert parse_paren_depth('(()())')==2\nassert parse_paren_depth('')==0\nassert parse_paren_depth('()()()')==1\nassert parse_paren_depth('((()))')==3\n"),
    ("is_palindrome",
     "def is_palindrome(text: str) -> bool:\n    \"\"\"Return True if text is a palindrome, ignoring case, spaces, and punctuation.\"\"\"\n",
     "assert is_palindrome('A man, a plan, a canal: Panama')==True\nassert is_palindrome('')==True\nassert is_palindrome('race a car')==False\nassert is_palindrome('No lemon, no melon')==True\n"),
    ("median",
     "from typing import List\n\ndef median(l: List[float]) -> float:\n    \"\"\"Return the median of elements in the list l (average of two middle values if even length).\"\"\"\n",
     "assert median([3,1,2,4,5])==3\nassert median([-10,4,6,1000,10,20])==15.0\nassert median([1,2])==1.5\n"),
    ("greatest_common_divisor",
     "def greatest_common_divisor(a: int, b: int) -> int:\n    \"\"\"Return the greatest common divisor of two integers a and b.\"\"\"\n",
     "assert greatest_common_divisor(3,5)==1\nassert greatest_common_divisor(25,15)==5\nassert greatest_common_divisor(0,5)==5\nassert greatest_common_divisor(48,36)==12\n"),
    ("sum_to_n_safe",
     "def sum_to_n_safe(n: int) -> int:\n    \"\"\"Return the sum of integers from 1 to n inclusive. For n<=0 return 0.\"\"\"\n",
     "assert sum_to_n_safe(5)==15\nassert sum_to_n_safe(0)==0\nassert sum_to_n_safe(-3)==0\nassert sum_to_n_safe(1)==1\nassert sum_to_n_safe(100)==5050\n"),
]

def call(prefix, prompt):
    body = json.dumps({
        "model": MODEL, "max_tokens": 1024,
        "messages": [{"role": "user", "content":
            f"{prefix}Complete the following Python function. Return ONLY the full function in a single python code block.\n\n{prompt}"}],
    }).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)["content"][0]["text"]

def extract(txt):
    m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S)
    return m.group(1) if m else txt

def run_test(code, test):
    ns = {}
    try:
        exec(code, ns)
        exec(test, ns)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

results = {c: {"pass": 0, "fail": [], "n": 0} for c in CONDITIONS}
for cname, prefix in CONDITIONS.items():
    for pname, prompt, test in PROBLEMS:
        try:
            code = extract(call(prefix, prompt))
            ok, err = run_test(code, test)
        except Exception as e:
            ok, err = False, f"API/{type(e).__name__}: {e}"
        results[cname]["n"] += 1
        if ok:
            results[cname]["pass"] += 1
        else:
            results[cname]["fail"].append(f"{pname} ({err})")
        print(f"  {cname:14} {pname:24} {'PASS' if ok else 'FAIL '+err}", flush=True)

print("\n=== PASS RATE (model: %s) ===" % MODEL)
for c, r in results.items():
    print(f"  {c:14} {r['pass']}/{r['n']}  " + ("" if not r["fail"] else "  miss: " + "; ".join(r["fail"])))
