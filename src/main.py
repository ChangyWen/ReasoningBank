import argparse
import numpy as np
from agent import Agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="gpt-5")
    args = parser.parse_args()

    agent = Agent(model_name=args.model)
    memory_bank = agent.memory_bank

    """testing chat"""
    print(agent.chat(prompt="What is the capital of France?"))
    """testing embedding"""
    embeddings = agent.embed(texts=[
        "What is the capital of France?",
        "What are the longitude and latitude of the capital of France?",
        "The capital of France is?",
        "The geo-coordinates of Paris are?"
    ])
    for i, embedding in enumerate(embeddings):
        print(f"embedding {i}:", np.array(embedding).shape, np.array(embedding).dtype, np.linalg.norm(np.array(embedding)))
    """testing memory bank"""
    memory_bank.add_memory(title="Capital of France", content="The capital of France is Paris.", embedding=embeddings[0])
    memory_bank.add_memory(title="Coordinates of France", content="Paris is in the north of France, with longitude 2.35 and latitude 48.85.", embedding=embeddings[1])
    print(memory_bank.get_memory(query_embedding=embeddings[2], top_k=2))
    print(memory_bank.get_memory(query_embedding=embeddings[3], top_k=2))

