import gradio as gr
import requests

# FastAPI backend endpoint
FASTAPI_URL = "http://127.0.0.1:8000/chat"


def ask_guru(message, chat_history):
    if not message:
        return chat_history, ""

    if chat_history is None:
        chat_history = []

    try:
        response = requests.post(
            FASTAPI_URL,
            json={"query": message}
        )
        response.raise_for_status()
        answer = response.json().get("answer", "No answer from backend")

    except Exception as e:
        answer = f" Error: {e}"

    # Append messages in Chatbot format
    chat_history.append((message, answer))

    return chat_history, ""  # clear input box


with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("## 💬 Ask Guru")

    chatbot = gr.Chatbot(
        height=500,
        show_label=False
    )

    msg = gr.Textbox(
        placeholder="Ask something from your documents...",
        lines=1
    )

    send_btn = gr.Button("Send")

    send_btn.click(
        ask_guru,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    msg.submit(
        ask_guru,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

demo.launch()
