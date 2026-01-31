# ask_guru
RAG based chatbot who will answer all your career related queries.

PDFs → Chunking → Embeddings → Qdrant
                                ↑
User → Gradio UI → FastAPI → Retriever → LLM → Answer
