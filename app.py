
from main import AppModel
import streamlit as st

MODEL_HF = "TheBloke/Mistral-7B-Code-16K-qlora-GGUF" # "TheBloke/Mistral-7B-OpenOrca-GGUF" 
#MODEL_HF = "TheBloke/Mistral-7B-Code-16K-qlora-GGUF" # "TheBloke/Mistral-7B-Instruct-v0.1-GGUF" # "TheBloke/Mistral-7B-OpenOrca-GGUF" 
MODEL_FILE = "mistral-7b-code-16k-qlora.Q4_K_M.gguf" # "mistral-7b-instruct-v0.1.Q4_K_M.gguf" # "mistral-7b-openorca.Q4_K_M.gguf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

pre_prompt_instruction = """
Chain of Thought: Process the information thoroughly. Understand the user's query in its entirety before formulating a response. Think step-by-step, ensuring a logical flow in the conversation.
Positivity: Maintain a friendly and positive demeanor throughout the conversation. Even in challenging situations, approach problems with a solution-oriented mindset.
Confidentiality: Respect user privacy. Do not ask for or disclose sensitive information. If users share sensitive data, avoid acknowledging it and gently guide the conversation to a safer topic.
Safety First: Prioritize the safety and well-being of users and others. Refrain from providing instructions that could cause harm or pose a risk.
"""

llm_response = ""


# Create a Streamlit app
st.title("ChatBot")

# init app
new_app = AppModel(EMBEDDING_MODEL, MODEL_HF, MODEL_FILE)

# Get User Prompt
input_prompt_box = st.text_input("Enter a prompt: ")


def query_llm(input_prompt):
    global pre_prompt_instruction, new_app
    last_msgs = str(new_app.chat_log[-3:])
    embed_result = new_app.get_embedding_docs(last_msgs + " \n\n " + input_prompt)[:new_app.context_limit]
    new_query = f"[Instruction]: {pre_prompt_instruction} \n\n [Data]: {str(embed_result)} \n\n "
    new_query += f"[Previous User Chat]: \n {last_msgs} \n\n [User Prompt]: \n {input_prompt} \n\n "
    new_response = new_app.get_llm_query(new_query)
    return new_response



# Initialize the Streamlit app
if st.button("Get LLM Query"):
    llm_response = query_llm(input_prompt_box)
    print(llm_response)
    st.write(llm_response)

    if st.button("Like & Save"):
        add_feedback(True)
        print("Feedback submitted")
        st.write("Feedback submitted")
        
    if st.button("Unlike & Discard"):
        add_feedback(False)
        print("Feedback submitted")
        st.write("Feedback submitted")
        

#if __name__ == "__main__":
#    pass

