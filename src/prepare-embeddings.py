import argparse
import json
import numpy as np
from typing import List, Tuple
from agent import Agent
import pandas as pd


subfix = """
Please reason step by step and then provide the final answer. The reasoning process must be enclosed within <think> </think> tags. The final answer MUST be put in \\boxed{}.
At any point during your reasoning, if you become highly unsure, or find the problem unsolvable or beyond your capability, stop attempting a solution.
In this case, instead of generating a possibly incorrect solution or guessing an answer, just honestly output \\boxed{Cannot Solve} with some brief explanation.
""".strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="gpt-5")
    parser.add_argument("--top_k", type=int, required=False, default=10)
    args = parser.parse_args()

    agent = Agent(model_name=args.model, top_k=args.top_k)

    files = [
        "data/omni_math_rule/omni_math_rule.parquet",
        "data/math/aime.parquet", # 30
        "data/math/amc.parquet", # 83
        "data/math/minerva.parquet", # 272
        "data/math/olympiad_bench.parquet", # 675
        "data/math/math-500.parquet", # 500
        "data/gsm8k/test.parquet", # 1319
    ]

    for file in files:
        data = pd.read_parquet(file)

        for idx, row in data.iterrows():
            # print(row)
            # print(row["extra_info"])
            # print(row["extra_info"]["embedding"])
            # print(np.array(row["extra_info"]["embedding"]).shape)
            # print(np.array(row["extra_info"]["embedding"]).dtype)
            # print(np.linalg.norm(np.array(row["extra_info"]["embedding"])))
            # input()
            prompt = row["prompt"][0]["content"]
            assert prompt.endswith(subfix), f"prompt={prompt}"
            raw_question = prompt.replace(subfix, "").strip()
            embedding = agent.embed(texts=[raw_question])[0]
            extra_info = row["extra_info"]
            extra_info["embedding"] = embedding
            data.at[idx, "extra_info"] = extra_info
            print(extra_info)
            print("embedding prepared for", idx)

        data.to_parquet(file.replace(".parquet", "_with_embedding.parquet"))
