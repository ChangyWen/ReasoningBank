import argparse
import json
import numpy as np
from typing import List, Tuple
from agent import Agent


def evaluate_response(response: str, ground_truth: str) -> float:
    pass


def extract_memory(response: str) -> Tuple[str, str]:
    pass


def get_task_prompt(question: str, memory_items: List[Tuple[str, str]]) -> str:
    pass


def get_memory_prompt(question: str, response: str, solution: str, score: float) -> str:
    pass


def load_dataset(dataset_file: str) -> Tuple[List[np.array], List[str], List[str], List[str]]:
    embeddings = []
    questions = []
    solutions = []
    ground_truths = []

    with open(dataset_file, "r") as f:
        for line in f:
            item = json.loads(line)
            embedding = np.array(item["embedding"]).astype(np.float32).reshape(1, -1)
            question = item["question"]
            solution = item["solution"]
            ground_truth = item["ground_truth"]

            embeddings.append(embedding)
            questions.append(question)
            solutions.append(solution)
            ground_truths.append(ground_truth)

    return embeddings, questions, solutions, ground_truths


def agent_training(agent: Agent, dataset_file: str):
    embeddings, questions, solutions, ground_truths = load_dataset(dataset_file)
    for embedding, question, solution, ground_truth in zip(embeddings, questions, solutions, ground_truths):
        relevant_memory_items = agent.get_memory(query_embedding=embedding)

        task_prompt = get_task_prompt(question=question, memory_items=relevant_memory_items)
        task_response = agent.chat(prompt=task_prompt)
        score = evaluate_response(response=task_response, ground_truth=ground_truth)

        memory_prompt = get_memory_prompt(question=question, response=task_response, solution=solution, score=score)
        memory_response = agent.chat(prompt=memory_prompt)
        memory_title, memory_content = extract_memory(response=memory_response)

        agent.add_memory(title=memory_title, content=memory_content, embedding=embedding)


def agent_testing(agent: Agent, dataset_file: str):
    embeddings, questions, solutions, ground_truths = load_dataset(dataset_file)
    for embedding, question, solution, ground_truth in zip(embeddings, questions, solutions, ground_truths):
        relevant_memory_items = agent.get_memory(query_embedding=embedding)

        task_prompt = get_task_prompt(question=question, memory_items=relevant_memory_items)
        task_response = agent.chat(prompt=task_prompt)
        score = evaluate_response(response=task_response, ground_truth=ground_truth)

    # process all scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="gpt-5")
    parser.add_argument("--top_k", type=int, required=False, default=10)
    args = parser.parse_args()

    agent = Agent(model_name=args.model, top_k=args.top_k)

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
    agent.add_memory(title="Capital of France", content="The capital of France is Paris.", embedding=embeddings[0])
    agent.add_memory(title="Coordinates of France", content="Paris is in the north of France, with longitude 2.35 and latitude 48.85.", embedding=embeddings[1])
    print(agent.get_memory(query_embedding=embeddings[2]))
    print(agent.get_memory(query_embedding=embeddings[3]))
