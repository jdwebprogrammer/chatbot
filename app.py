
from main import AppModel
import gradio as gr
from gradio.components import Markdown, Textbox, Button

MODEL_HF = "TheBloke/Mistral-7B-Code-16K-qlora-GGUF" # "TheBloke/Mistral-7B-OpenOrca-GGUF"  "TheBloke/Mistral-7B-Code-16K-qlora-GGUF" # "TheBloke/Mistral-7B-Instruct-v0.1-GGUF" # "TheBloke/Mistral-7B-OpenOrca-GGUF" 
MODEL_FILE = "mistral-7b-code-16k-qlora.Q4_K_M.gguf" # "mistral-7b-instruct-v0.1.Q4_K_M.gguf" # "mistral-7b-openorca.Q4_K_M.gguf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

pre_prompt_instruction = """
Chain of Thought: Process the information thoroughly. Understand the user's query in its entirety before formulating a response. Think step-by-step, ensuring a logical flow in the conversation.
Positivity: Maintain a friendly and positive demeanor throughout the conversation. Even in challenging situations, approach problems with a solution-oriented mindset.
Confidentiality: Respect user privacy. Do not ask for or disclose sensitive information. If users share sensitive data, avoid acknowledging it and gently guide the conversation to a safer topic.
Safety First: Prioritize the safety and well-being of users and others. Refrain from providing instructions that could cause harm or pose a risk.
"""

llm_response = ""
history = []

# init app
new_app = AppModel(EMBEDDING_MODEL, MODEL_HF, MODEL_FILE, context_limit=5000, context_length=16000)



def query_llm(input_prompt, new_history):
    global history, pre_prompt_instruction, new_app
    history = new_history
    last_msgs = str(new_app.chat_log[-3:])
    embed_result = new_app.get_embedding_docs(last_msgs + " \n\n " + input_prompt)[:new_app.context_limit]
    new_query = f"[Instruction]: {pre_prompt_instruction} \n\n [Data]: {str(embed_result)} \n\n "
    new_query += f"[Previous User Chat]: \n {last_msgs} \n\n [User Prompt]: \n {input_prompt} \n\n "
    new_response = new_app.get_llm_query(new_query)
    return new_response



def feedback_like():
    new_app.add_feedback(True)
    print("Feedback submitted")
    gr.Info("Feedback submitted")

def feedback_dislike():
    new_app.add_feedback(False)
    print("Feedback submitted")
    gr.Info("Feedback submitted")


with gr.Blocks(title="ChatBot", analytics_enabled=False) as chatbot:
    gr.Markdown("# ChatBot")
    gr.Markdown("Welcome to ChatBot!")
    with gr.Row():
        with gr.Column(scale=1):
            gr.ChatInterface(query_llm)
    with gr.Row():
        with gr.Column(scale=1):
            feedback_btn_like = gr.Button(value="Like & Save")
        with gr.Column(scale=1):
            feedback_btn_dislike = gr.Button(value="Dislike & Discard")
        feedback_btn_like.click(fn=feedback_like)
        feedback_btn_dislike.click(fn=feedback_dislike)

chatbot.queue().launch(server_name="0.0.0.0", server_port=7864, show_api=False)