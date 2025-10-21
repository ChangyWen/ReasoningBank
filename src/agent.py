import os
import json
import argparse
from typing import List, Optional, Callable, Tuple, Dict
from tqdm import tqdm
from datasets import Dataset
from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.core.credentials import TokenCredential
from azure.core.pipeline.policies import BearerTokenCredentialPolicy
from azure.core.pipeline import PipelineRequest, PipelineContext
from azure.core.rest import HttpRequest
import threading
import functools
import re
import time
from uuid import uuid4
import numpy as np
from memory import MemoryBank


class Agent:
    def __init__(self, model_name: str):
        self.model_name = model_name

        azure_client = self.load_azure_client()

        if model_name in ["gpt-5", "gpt-4.1-nano"]:
            self.model = azure_client
        else:
            self.model = self.load_vllm_client()

        self.embedder = azure_client

        self.memory_bank = MemoryBank()


    def _make_request(self) -> PipelineRequest[HttpRequest]:
        return PipelineRequest(
            HttpRequest("CredentialWrapper", "https://fakeurl"), PipelineContext(None)
        )


    def get_bearer_token_provider(
        credential: TokenCredential, *scopes: str
    ) -> Callable[[], str]:
        policy = BearerTokenCredentialPolicy(credential, *scopes)

        def wrapper(self) -> str:
            request = self._make_request()
            policy.on_request(request)
            return request.http_request.headers["Authorization"][len("Bearer ") :]

        return wrapper


    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(
            managed_identity_client_id="e6162a0d-e540-4454-995f-30bcb97f35b4"
        ),
        "https://cognitiveservices.azure.com/.default",
    )


    def load_azure_client(self):
        return AzureOpenAI(
            api_version="2025-01-01-preview",
            azure_endpoint="https://csnf-singularity-aoai-eastus2.openai.azure.com/",
            azure_ad_token_provider=self.token_provider,
        )


    def load_vllm_client(self):
        return OpenAI(base_url="http://localhost:8000/v1", api_key="empty")


    """
    Main functions: Embedding
    """
    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.embedder.embeddings.create(input=texts, model="csnf-text-embedding-3-large")
        return [item.embedding for item in response.data]


    """
    Main functions: Chat
    """
    def chat(self, prompt: str) -> str:
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2048,
        )
        return response.choices[0].message.content
