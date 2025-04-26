# prep_data.py
import json, pathlib

SRC = pathlib.Path("few_shot_examples.json")   # your uploaded file
DST = pathlib.Path("train.jsonl").open("w")

raw = json.loads(SRC.read_text())

for row in raw:
    user_q = row["text"].strip()
    answer  = "### FILL_ME ###"               # placeholder for now
    obj = {
        "messages": [
            {"role": "system",
             "content": "You are a helpful assistant."},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": answer}
        ]
    }
    DST.write(json.dumps(obj) + "\n")

print("Wrote", len(raw), "lines to train.jsonl")
