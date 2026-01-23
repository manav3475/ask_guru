import torch

from typing import List
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from logger import logger


# Load once (IMPORTANT: do NOT load inside function repeatedly)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

logger.info(f"Loading LLM model: {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)


text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.0,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)


def generate_answer(
    query: str,
    retrieved_docs: List[Document],
) -> str:
    """
    Generate an answer using an LLM based on retrieved documents.
    """

    # Combine retrieved documents into context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    logger.debug(f"Context for generation: {context}")


    # Build prompt
    prompt = f"""
You are an AI assistant.
Answer the question ONLY using the context below.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

    logger.debug(f"Prompt for generation: {prompt}")

    return llm(prompt).strip()
