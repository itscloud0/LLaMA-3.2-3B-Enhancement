from datasets import load_dataset
import pandas as pd
import random

def load_flan_sampled(sample_size=60000, chunk_count=100):
    print("Loading FLAN v2 (streaming)...")
    flan_iter = load_dataset("SirNeural/flan_v2", split="train", streaming=True)

    print(f"Dividing into {chunk_count} chunks...")
    chunks = [[] for _ in range(chunk_count)]
    for i, example in enumerate(flan_iter):
        chunks[i % chunk_count].append(example)
        if i >= sample_size * 2:  # early stop for sampling
            break

    print("Sampling from each chunk...")
    sampled = []
    per_chunk = sample_size // chunk_count
    for chunk in chunks:
        random.shuffle(chunk)
        selected = chunk[:per_chunk]
        for ex in selected:
            if ex.get("inputs") and ex.get("targets"):
                sampled.append({
                    "instruction": ex["inputs"],
                    "input": "",
                    "output": ex["targets"]
                })

    return sampled

def load_gsm8k():
    print("Loading GSM8K...")
    gsm = load_dataset("gsm8k", "main", split="train")
    return [
        {"instruction": ex["question"], "input": "", "output": ex["answer"]}
        for ex in gsm if ex.get("question") and ex.get("answer")
    ]

def main():
    flan_data = load_flan_sampled()
    gsm_data = load_gsm8k()

    combined = flan_data + gsm_data
    print(f"Total combined examples: {len(combined)}")

    df = pd.DataFrame(combined)
    df.to_csv("singularANYlite.csv", index=False)
    print("Saved to singularANYlite.csv")

if __name__ == "__main__":
    main()
    print("Loading datasets...")
