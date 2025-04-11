import os
from langdetect import detect
from deep_translator import GoogleTranslator

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Step 2: Language Detection & Translation Functions

def translate_to_english(query):
    """Detects language & translates to English if needed."""
    detected_lang = detect(query)
    if detected_lang != "en":
        translated_query = GoogleTranslator(source=detected_lang, target="en").translate(query)
        print(f" Translated ({detected_lang} → en): {translated_query}")
        return translated_query, detected_lang
    return query, "en"

def translate_response(response, target_lang):
    """Translates response back to user's original language."""
    if target_lang != "en":
        return GoogleTranslator(source="en", target=target_lang).translate(response)
    return response

# Step 3: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say that you don’t know. Don’t try to make up an answer.
Only use the given context, do not provide any additional information.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 4: Get User Query & Process It
user_query = input("Write Query Here: ")

# Translate input if necessary
translated_query, original_lang = translate_to_english(user_query)

# Retrieve answer
response = qa_chain.invoke({'query': translated_query})

# Translate response back if needed
final_response = translate_response(response["result"], original_lang)

# Display Results
print("\n🔹 RESULT: ", final_response)
print("\n📄 SOURCE DOCUMENTS: ", response["source_documents"])
