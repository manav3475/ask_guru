import torch

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.logger import logger


# Load once (IMPORTANT: do NOT load inside function repeatedly)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

logger.debug(f"Preparing LLM model...".format(MODEL_NAME))

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


def generate_answer(query: str, retrieved_docs):
    if not retrieved_docs:
        return "I could not find relevant information."
        logger.warning("No documents retrieved for the query.") 

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
        if isinstance(doc.page_content, str)
    )

    if not context.strip():
        return "I found documents, but they contain no readable text."
        logger.warning("Retrieved documents contain no readable text.")

    prompt = f"""
    You are an AI assistant.

    Use the context below to answer the question.
    If the answer is not fully present, answer as best as you can
    based on the context. If the context is completely irrelevant,
    say "I don't know".

    Context:
    {context}

    Question:
    {query}

    Answer in clear, simple language:
    """

    logger.debug(f"Generated prompt: {prompt}")

    return llm.invoke(prompt).strip()
