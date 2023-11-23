import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG)

from main import AppModel
import gradio as gr
from gradio.components import Markdown, Textbox, Button

pre_prompt_instruction = """
Chain of Thought: Process the information thoroughly. Understand the user's query in its entirety before formulating a response. Think step-by-step, ensuring a logical flow in the conversation.
Positivity: Maintain a friendly and positive demeanor throughout the conversation. Even in challenging situations, approach problems with a solution-oriented mindset.
Confidentiality: Respect user privacy. Do not ask for or disclose sensitive information. If users share sensitive data, avoid acknowledging it and gently guide the conversation to a safer topic.
Safety First: Prioritize the safety and well-being of users and others. Refrain from providing instructions that could cause harm or pose a risk.
"""

llm_response = ""
history = []

# init app
new_app = AppModel()



def query_llm(input_prompt, new_history):
    global history, pre_prompt_instruction, new_app
    history = new_history
    last_msgs = str(new_app.chat_log[-3:])
    embed_result = new_app.get_embedding_docs(last_msgs + " \n\n " + input_prompt)[:new_app.context_limit]
    new_query = f"Instruction: {pre_prompt_instruction} \n\n Retrieved Context: {str(embed_result)} \n\n "
    #new_query += f"Wiki Results: \n {new_app.search_wiki(input_prompt)} \n\n " 
    new_query += f"Previous User Chat: \n {last_msgs} \n\n User Prompt: \n {input_prompt} \n\n AI Response: \n "
    new_response = new_app.get_llm_query(new_query, input_prompt)
    return new_response



def feedback_like():
    new_app.add_feedback(True)
    logging.info("Feedback submitted")
    gr.Info("Feedback submitted")

def feedback_dislike():
    new_app.add_feedback(False)
    logging.info("Feedback submitted")
    gr.Info("Feedback submitted")


with gr.Blocks(title="ChatBot", analytics_enabled=False) as chatbot:
    gr.Markdown("# ChatBot")
    gr.Markdown("Welcome to ChatBot!")
    with gr.Row():
        with gr.Column(scale=1):
            gr.ChatInterface(query_llm, examples=[
                "What is today's date?", 
                "Explain the limitations of natural language processing in current AI systems.", 
                "Compose a poem about the beauty of nature.",
                "Write a Python function to calculate the factorial of a number.",
                "How would you solve the traveling salesman problem using a heuristic algorithm?"], analytics_enabled=False)
    with gr.Row():
        with gr.Column(scale=1):
            feedback_btn_like = gr.Button(value="Like & Save")
        with gr.Column(scale=1):
            feedback_btn_dislike = gr.Button(value="Dislike & Discard")
        feedback_btn_like.click(fn=feedback_like)
        feedback_btn_dislike.click(fn=feedback_dislike)

chatbot.queue().launch(server_name="0.0.0.0", server_port=7864, show_api=False)