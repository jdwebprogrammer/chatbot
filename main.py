
from ctransformers import AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from pathlib import Path
import chromadb
import os
import json

# TheBloke/deepseek-coder-33B-instruct-GGUF  "ddh0/Yi-6B-200K-GGUF-fp16" 
#  "TheBloke/Mistral-7B-Code-16K-qlora-GGUF" # "TheBloke/Mistral-7B-Instruct-v0.1-GGUF" # "TheBloke/Mistral-7B-OpenOrca-GGUF" 
# "NousResearch/Yarn-Mistral-7b-128k" "JDWebProgrammer/custom_sft_adapter" 

MODEL_HF = "TheBloke/Yarn-Mistral-7B-128k-GGUF"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class AppModel:
    def __init__(self, embedding_model_name=EMBEDDING_MODEL, model=MODEL_HF, dataset_path="./data/logs", dir="./data", 
        context_limit=32000, temperature=0.8, max_new_tokens=4096, context_length=128000):
        self.model = model
        self.embedding_model_name = embedding_model_name
        self.model_config = AutoConfig.from_pretrained(self.model, context_length=context_length)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_name.split("/")[1])
        self.chroma_client = chromadb.PersistentClient(path="./data/vectorstore", settings=Settings(anonymized_telemetry=False))
        self.sentences = [] 
        self.ref_collection = self.chroma_client.get_or_create_collection("ref", embedding_function=self.emb_fn)
        self.logs_collection = self.chroma_client.get_or_create_collection("logs", embedding_function=self.emb_fn)
        self.init_chroma()

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(self.model, model_type="mistral", config=self.model_config) #, cache_dir="./models" , gpu_layers=0 local_files_only=True) , 
        self.chat_log = []
        self.last_ai_response = ""
        self.last_user_prompt = "" 
        self.context_limit=context_limit
        self.temperature=temperature
        self.max_new_tokens=max_new_tokens

    def get_llm_query(self, input_prompt, user_prompt):
        self.last_user_prompt = str(user_prompt)
        new_response = self.llm(prompt=input_prompt, temperature=self.temperature, max_new_tokens=self.max_new_tokens) #, temperature=self.temperature, max_new_tokens=self.max_new_tokens)
        self.last_ai_response = str(new_response)
        self.save_file(f"[User_Prompt]: {user_prompt} \n[AI_Response]: {new_response} \n", "./data/logs/chat-log.txt")
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

    def build_chroma_docs(self, directory="./data/context", id_name="doc_", metatag={"source": "notion"}):
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
                anum = 0
                for a in file_contents.split(splitter): # split first by paragraph  
                    docs.append(a)
                    ids.append(id_name + str(fnum))
                    additional_metas = {"dir": directory, "filename":file_path, "chunk_number": anum }
                    metas.append({**metatag, **additional_metas})
                    fnum += 1
                    anum += 1
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
        feedback_str = ""
        if is_positive:
            feedback_str = "GOOD/PASS"
            self.chat_log.append(self.last_ai_response[:self.context_limit])
            self.save_file(self.last_ai_response)
        else:
            feedback_str = "BAD/FAIL"
        new_obj = f"[User_Prompt]: {self.last_user_prompt}\n[AI_Response]: {self.last_ai_response}\n[User_Feedback]: {feedback_str}\n\n"
        self.save_file(new_obj, "./data/logs/feedback-log.txt")


    def open_file(self, file_path):
        file_contents = ""
        with open(file_path, "r") as file:
            file_contents = file.read()
        return file_contents



