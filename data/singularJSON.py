# This script loads the Dolly, OpenAssistant, and Alpaca datasets, processes them, and saves them to a JSON file.
# It is designed to be run in a Python environment with the required libraries installed.
from datasets import load_dataset
import requests
import json

def load_dolly():
    print("Loading Dolly dataset...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    return [
        {
            "instruction": ex["instruction"],
            "input": ex["context"] or "",
            "output": ex["response"]
        }
        for ex in ds
    ]

def load_oasst():
    print("Loading OpenAssistant dataset...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    return [
        {
            "instruction": ex.get("prompt", ""),
            "input": "",
            "output": ex["text"]
        }
        for ex in ds
        if ex["role"] == "assistant" and ex.get("rank") == 0 and ex.get("text")
    ]

def load_alpaca():
    print("Downloading Alpaca dataset...")
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    response = requests.get(url)
    raw_data = response.json()
    return [
        {
            "instruction": ex["instruction"],
            "input": ex["input"],
            "output": ex["output"]
        }
        for ex in raw_data
    ]

def main():
    dolly_data = load_dolly()
    oasst_data = load_oasst()
    alpaca_data = load_alpaca()

    print("Merging datasets...")
    all_data = dolly_data + oasst_data + alpaca_data

    print(f"Total examples: {len(all_data)}")
    with open("singularJSON.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print("Saved merged dataset to singularJSON.json")

if __name__ == "__main__":
    main()
    print("Loading datasets...")