# THIS FILE WILL GENERATE AN 8GB FILE SO DON'T RUN IT UNLESS YOU HAVE ENOUGH SPACE
# This script loads the FLAN v2 and GSM8K datasets, processes them, and saves them to a CSV file.
# It is designed to be run in a Python environment with the required libraries installed.
from datasets import load_dataset
import pandas as pd

def load_flan():
    print("Loading FLAN v2...")
    flan = load_dataset("SirNeural/flan_v2", split="train")
    flan_data = [
        {"instruction": ex["inputs"], "input": "", "output": ex["targets"]}
        for ex in flan if ex.get("inputs") and ex.get("targets")
    ]
    return flan_data

def load_gsm8k():
    print("Loading GSM8K...")
    gsm = load_dataset("gsm8k", "main", split="train")
    gsm_data = [
        {"instruction": ex["question"], "input": "", "output": ex["answer"]}
        for ex in gsm if ex.get("question") and ex.get("answer")
    ]
    return gsm_data

def main():
    flan_data = load_flan()
    gsm_data = load_gsm8k()

    combined = flan_data + gsm_data
    print(f"Total combined examples: {len(combined)}")

    df = pd.DataFrame(combined)
    df.to_csv("singularANY.csv", index=False)
    print("Saved to singularANY.csv")

if __name__ == "__main__":
    main()
    print("Loading datasets...")