import gradio as gr
import requests

FASTAPI_URL = "http://127.0.0.1:8000/chat"

def ask_guru(message, chat_history):
    if not message:
        return chat_history, ""

    if chat_history is None:
        chat_history = []

    try:
        
        response = requests.post(
                    FASTAPI_URL,
                    json={"query": message},
                    timeout=120
        )

        response.raise_for_status()
        answer = response.json().get("answer", "No answer from backend")
    except Exception as e:
        answer = f" Error: {e}"

    chat_history.append(
        {"role": "user", "content": message}
    )
    chat_history.append(
        {"role": "assistant", "content": answer}
    )

    return chat_history, ""

with gr.Blocks() as demo:
    gr.Markdown("## 💬 Ask Guru")

    chatbot = gr.Chatbot(
        height=500,
        type="messages"   # 🔑 REQUIRED
    )

    msg = gr.Textbox(
        placeholder="Ask something from your documents...",
        lines=1
    )

    send_btn = gr.Button("Send")

    send_btn.click(ask_guru, [msg, chatbot], [chatbot, msg])
    msg.submit(ask_guru, [msg, chatbot], [chatbot, msg])

demo.launch(share=True)
