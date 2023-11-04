
from ctransformers import AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from pathlib import Path
import joblib

import chromadb
import os
import json



class AppModel:
    def __init__(self, embedding_model_name, model, model_file, dataset_path="./data/logs", dir="./data", context_limit=400, temperature=0.8, max_new_tokens=1024, context_length=1024):
        self.model = model
        self.model_file = model_file
        self.embedding_model_name = embedding_model_name
        self.model_config = AutoConfig.from_pretrained(self.model, context_length=context_length, allow_reset=True)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_name.split("/")[1])
        self.chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",anonymized_telemetry=False)) #,persist_directory="./data/vectorstore"
        self.chroma_client.reset()
        self.sentences = [] 
        self.ref_collection = self.chroma_client.get_or_create_collection("ref", embedding_function=self.emb_fn)
        self.logs_collection = self.chroma_client.get_or_create_collection("logs", embedding_function=self.emb_fn)
        self.init_chroma()

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(self.model, model_type="mistral", config=self.model_config) # , gpu_layers=0 local_files_only=True) cache_dir="./models", 
        self.chat_log = []
        self.last_ai_response = ""
        self.last_user_prompt = "" 
        self.context_limit=context_limit
        self.temperature=temperature
        self.max_new_tokens=max_new_tokens

    def get_llm_query(self, input_prompt):
        self.last_user_prompt = str(input_prompt)
        new_response = self.llm(prompt=input_prompt) #, temperature=self.temperature, max_new_tokens=self.max_new_tokens)
        self.last_ai_response = str(new_response)
        self.chat_log.append(new_response[:self.context_limit])
        self.save_file(f"USER: {input_prompt} \nAI_RESPONSE: {new_response} \n", "./data/logs/chat-log.txt")
        return new_response
        
    def get_embedding_values(self, input_str):
        tokenized_input = self.build_embeddings(input_str)
        print(tokenized_input)
        embedding_values = self.embedding_model.encode(tokenized_input)
        return embedding_values

    def get_embedding_docs(self, query_text, n_results=2):
        query_embeddings = self.get_embedding_values(query_text).tolist()[0]
        query_result = self.ref_collection.query(query_embeddings=query_embeddings,n_results=n_results) 
        return query_result["documents"]

    def init_chroma(self):
        docs, metas, ids = self.build_chroma_docs(directory="./data/reference", id_name="ref_")
        if docs:
            print(f"Loading Chroma (Reference) Docs: {len(docs)}")
            self.ref_collection.add(documents=docs, metadatas=metas, ids=ids)

        docs, metas, ids = self.build_chroma_docs(directory="./data/context", id_name="context_")
        if docs:
            print(f"Loading Chroma (Context) Docs: {len(docs)}")
            self.logs_collection.add(documents=docs, metadatas=metas, ids=ids)

    def build_chroma_docs(self, directory="./data/context", id_name="doc_"):
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

    def save_file(self, data, filename="./data/context/chat-log.txt"):
        with open(filename, 'a') as f:
            f.write('\n\n' + str(data))
    
    def add_feedback(self, is_positive=True):
        new_obj = { "user_prompt": self.last_user_prompt, "ai_response": self.last_ai_response, "is_positive": is_positive }
        if is_positive:
            self.save_file(self.last_ai_response)
        self.save_file(new_obj, "./data/logs/feedback-log.txt")
