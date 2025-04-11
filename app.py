import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langdetect import detect
from deep_translator import GoogleTranslator
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# Load Vectorstore (FAISS)
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    # initially used sentence-transformers/all-MiniLM-L6-v2 
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Setup Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the provided context to answer the user's question.
If the answer is unknown, say you don't know. Don't guess.
If exact keywords are not mentioned, infer based on related documentation if possible.
Provide only relevant information from the given context.

Context: {context}
Question: {question}

Start the answer directly. No extra comments.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load LLM
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Translation Functions
def detect_and_translate_to_english(query):
    """Detects language & translates to English if needed."""
    detected_lang = detect(query)
    if detected_lang != "en":
        translated_query = GoogleTranslator(source=detected_lang, target="en").translate(query)
        return translated_query, detected_lang
    return query, "en"

def translate_response(response, target_lang):
    """Translates response back to the user's original language."""
    if target_lang != "en":
        return GoogleTranslator(source="en", target=target_lang).translate(response)
    return response

def extract_pdf_names(source_documents):
    pdf_names = []
    for doc in source_documents:
        source_path = doc.metadata.get("source", " ")
        filename = os.path.basename(source_path)
        pdf_names.append(filename)
    return list(set(pdf_names))  # removes duplicates if needed


# Main App
def main():
    st.title(" Multilingual AI Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask me anything in any language...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            # Detect & Translate Query
            translated_query, original_lang = detect_and_translate_to_english(prompt)

            # Load Vectorstore
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # Setup Retrieval Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 5}), # k = 3 is changed to 5 here
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Get Response
            response = qa_chain.invoke({'query': translated_query})
            result = response["result"]
            source_documents = response["source_documents"]

            # Translate response back to original language
            final_response = translate_response(result, original_lang)

            source_docs = extract_pdf_names(source_documents)

            result_to_show = final_response + "\n\n**Source Docs:**\n" + str(source_docs)

            # Display Response
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': final_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
