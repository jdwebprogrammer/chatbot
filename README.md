
# ChatBot
Basic ChatBot using CTransformers, ChromaDB and Gradio
- See a live demo on HuggingFace at:
- https://huggingface.co/spaces/JDWebProgrammer/chatbot

![ChatBot](./assets/chatbot.png "ChatBot")

# Experimental
Please note that AI is an area of thorough research with known problems such as bias, misinformation and leaking sensitive information. We cannot guarantee the accuracy, completeness, or timeliness of the information provided. We do not assume any responsibility or liability for the use or interpretation of this project.

# Description
This is a simple ChatBot to use as a simple starting template. Just add text files into the "./data/reference" folder 
![ChatBot Logic](./assets/logic.png "ChatBot Logic")

# Features
- Full custom RAG implementation
- Copy text files into ./data/reference for embedding
- Auto save chat logs
- Auto download and run open source LLM's locally
- Currently using the awesome combined works of Mistral AI's LLM base model trained with 128k context window by NousResearch and quantized to 4bits for fast speed by TheBloke

# Step 1: Install Dependencies
First make sure you have python and pip installed. Then open a terminal and type:
```shell
pip install -r requirements.txt
```


# Step 2: Add Embeddings [Optional]
Place text files in "./data/reference" to enhance the chatbot with extra information

# Step 3: Run Chatbot
Open a terminal and type:
```shell
python app.py
```

The web interface will start at http://0.0.0.0:7864

# Progress & Updates
- Embeddings properly save & persist, full custom RAG implementation working

# Known Issues
- Chat history may not be saving properly! Working on this..
- There is currently no feedback but will save feedback logs

# Future Plans
- Will be implementing auto retrieval from wiki for RAG(Retrieval Augmented Generation) loop
- Feedback datasets will be used in the trainable project coming soon
- Still working on instructional syntax which could be causing impaired results

# Credits
- Mistral: https://mistral.ai/
- HuggingFace: https://huggingface.co/
- TheBloke: https://huggingface.co/TheBloke
- ctransformers: https://github.com/marella/ctransformers
- gradio: https://github.com/gradio-app/gradio
- chroma: https://github.com/chroma-core/chroma

