
from ctransformers import AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from pathlib import Path

import streamlit as st
import chromadb
import os
import json


MODEL_DIR = ""
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

pre_prompt_instruction = """
Chain of Thought: Process the information thoroughly. Understand the user's query in its entirety before formulating a response. Think step-by-step, ensuring a logical flow in the conversation.
Encouragement: Encourage users to experiment and learn. Support their curiosity and guide them effectively toward solutions.
Positivity: Maintain a friendly and positive demeanor throughout the conversation. Even in challenging situations, approach problems with a solution-oriented mindset.
Avoidance of Loops: If you sense a conversation loop, gently steer the discussion in a new direction. Offer alternative suggestions or seek clarification to break the cycle.
Confidentiality: Respect user privacy. Do not ask for or disclose sensitive information. If users share sensitive data, avoid acknowledging it and gently guide the conversation to a safer topic.
Safety First: Prioritize the safety and well-being of users and others. Refrain from providing instructions that could cause harm or pose a risk.
"""


class AppModel:
    def __init__(self, embedding_model_name=EMBEDDING_MODEL, dataset_path="./data/logs", dir="./data"):
        global MODEL_DIR
        st.write("Starting app")
        self.embedding_model_name = embedding_model_name
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False,persist_directory="./data/vectorstore"))
        #self.chroma_client_persist = chromadb.PersistentClient(path="./data/vectorstore",settings=Settings(anonymized_telemetry=False))
        self.new_model_name = "custom_model"
        self.new_model_path = f"./{self.new_model_name}"
        self.sentences = [] 
        self.ref_collection = self.chroma_client.get_or_create_collection("ref", embedding_function=self.emb_fn)
        self.logs_collection = self.chroma_client.get_or_create_collection("logs", embedding_function=self.emb_fn)
        self.init_chroma()

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)
        self.chat_log = []
        self.context_limit = 400
        self.temperature=0.2
        self.max_new_tokens=1024

    def get_llm_query(self, input_prompt):
        global pre_prompt_instruction
        last_msgs = str(self.chat_log[-3:])
        embed_result = self.get_embedding_docs(last_msgs + " \n\n " + input_prompt)[:self.context_limit]
        new_query = f"[INSTRUCTION]{pre_prompt_instruction}[/INSTRUCTION] \n\n [DATA]{str(embed_result)}[/DATA] \n\n "
        new_query += f"[Previous User Chat]: \n {last_msgs} \n\n [User Prompt]: \n {input_prompt} \n\n [Response]: \n "
        new_response = self.llm(prompt=new_query, temperature=self.temperature, max_new_tokens=self.max_new_tokens)
        self.chat_log.append(new_response[:self.context_limit])
        return new_response
        
    def get_embedding_values(self, input_str):
        tokenized_input = self.build_embeddings(input_str)
        print(tokenized_input)
        embedding_values = self.embedding_model.encode(tokenized_input)
        #print(embedding_values)
        return embedding_values

    def get_embedding_docs(self, query_text, n_results=2):
        query_embeddings = self.get_embedding_values(query_text).tolist()[0]
        query_result = self.ref_collection.query(query_embeddings=query_embeddings,n_results=n_results) 
        return query_result["documents"]

    def init_chroma(self):
        docs, metas, ids = self.build_chroma_docs(directory="./data/reference", id_name="ref_")
        if docs:
            print(f"Loading Chroma Docs: REF {len(docs)}")
            st.write(f"Loading Chroma Docs: REF {len(docs)}")
            self.ref_collection.add(documents=docs, metadatas=metas, ids=ids)

        docs, metas, ids = self.build_chroma_docs(directory="./data/logs", id_name="log_")
        if docs:
            print(f"Loading Chroma Docs: LOGS {len(docs)}")
            st.write(f"Loading Chroma Docs: LOGS {len(docs)}")
            self.logs_collection.add(documents=docs, metadatas=metas, ids=ids)

    def build_chroma_docs(self, directory="./data/logs", id_name="doc_"):
        directory = os.path.join(os.getcwd(), directory)
        docs = []
        metas = []
        ids = []
        fnum = 0
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                file_contents = file.read()
                splitter = "\n\n"
                if ".csv" in file_path:
                    splitter = "\n"
                for a in file_contents.split(splitter): # split first by paragraph  
                    docs.append(a)
                    ids.append(id_name + str(fnum))
                    metas.append({"source": "notion"})
                    fnum += 1
        docs = list(docs)
        metas = list(metas)
        ids = list(ids)
        return docs, metas, ids

    

    def build_embeddings(self, content, add_sentences=False):
        tokenized_sentences = []
        for b in content.split("\n"): # then by line 
            for c in b.split("  "): # then by tab
                for d in c.split(". "): # by sentence
                    tokenized_sentences.append(str(d))
                    if add_sentences:
                        self.sentences.append(str(d))
        return tokenized_sentences




# init app
new_app = AppModel()

# Create a Streamlit app
st.title("ChatBot")

# Get User Prompt
input_prompt_box = st.text_input("Enter a prompt: ")

# Initialize the Streamlit app
if st.button("Get LLM Query"):
    llm_response = new_app.get_llm_query(input_prompt_box)
    print(llm_response)
    st.write(llm_response)

