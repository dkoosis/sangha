import json, os, re, urllib.request
MODEL="claude-haiku-4-5-20251001"; KEY=os.environ["ANTHROPIC_API_KEY"]
CONDITIONS={"control":"","generic_hq":"Write high-quality, production-ready code.\n\n",
 "greek_arete":"Write code embodying ἀρετή (excellence).\n\n",
 "jp_shokunin":"Write code with 職人気質 (craftsman spirit).\n\n",
 "combined":"Write code embodying ἀρετή and 職人気質.\n\n"}
PROMPT=("def truncate_number(number: float) -> float:\n"
        "    \"\"\"Return the decimal part of a positive float.\"\"\"\n")
def call(p):
    body=json.dumps({"model":MODEL,"max_tokens":1024,"messages":[{"role":"user","content":
        f"{p}Complete the following Python function. Return ONLY the function in one python code block.\n\n{PROMPT}"}]}).encode()
    req=urllib.request.Request("https://api.anthropic.com/v1/messages",data=body,
        headers={"x-api-key":KEY,"anthropic-version":"2023-06-01","content-type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=90))["content"][0]["text"]
def code(t):
    m=re.search(r"```(?:python)?\s*(.*?)```",t,re.S); return (m.group(1) if m else t).strip()
for c,p in CONDITIONS.items():
    src=code(call(p)); lines=src.splitlines()
    print(f"\n===== {c} =====  lines={len(lines)}  chars={len(src)}  comments={sum('#' in l for l in lines)}  docstr={'\"\"\"' in src}")
    print(src)
