import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from transformers import AutoTokenizer, TextStreamer, pipeline, AutoModelForQuestionAnswering
from auto_gptq import AutoGPTQForCausalLM
import torch
from flask import Flask, jsonify,request
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
app = Flask(__name__)



index_name = 'chatbot-university'
    # Initialize Pinecone with your API key and environment
api_key = 'aa199580-6d1e-4a83-9668-44a3013f39f9'
pc = Pinecone(api_key=api_key)
index_name = 'chatbot-university'
    # Initialize Pinecone index
index = pc.Index(index_name)
from langchain_pinecone import PineconeVectorStore
#from pinecone_notebooks.colab import Authenticate
#Authenticate()
import os

pinecone_api_key =api_key #os.environ.get("PINECONE_API_KEY")
pinecone_api_key

import time

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)
print(pinecone_api_key)
import time

index_name = "chatbot-university"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
  print(f"Creating index {index_name}...")
    # pc.create_index(
    #     name=index_name,
    #     dimension=1536,
    #     metric="cosine",
    #     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    # )
    # while not pc.describe_index(index_name).status["ready"]:
    #     time.sleep(1)

index = pc.Index(index_name)
from langchain_pinecone import PineconeVectorStore
print(index)
embedding_model = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')
docsearch = PineconeVectorStore(index_name=index_name, embedding=embedding_model)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
model_basename = "model"
print(docsearch[0])
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    model_basename=model_basename,
    use_safetensors=True,
    trust_remote_code=True,
    inject_fused_attention=False,
    device=DEVICE,
    quantize_config=None,
)
model.eval()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,
    batch_size=5,
)
SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
def generate_prompt(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()

template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    verbose=False,
    retriever=docsearch.as_retriever(search_type="mmr"),
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt},
)
def clear_text_before_marker(text, marker):
   marker_position = text.find(marker)
   if marker_position != -1:
       return text[marker_position + len(marker):].strip()
   return text

#cleaned_output = clear_text_before_marker(result1['result'], "[/INST]")
#print(cleaned_output)
@app.route("/generate", methods=['POST'])
def generateresult(query):
    query = request.json.get('query', '')
    result = qa_chain(query)
    cleaned_output = clear_text_before_marker(result['result'], "[/INST]")
#    cleaned_result = result.replace("[INST] <>", "").strip()
    return jsonify({'result': cleaned_output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)























