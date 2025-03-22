import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import fitz
import tiktoken
import numpy as np

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

st.set_page_config(page_title="AI Text Assistant", layout="wide")
st.title("🧠 AI Text Assistant")

def split_into_chunks(content, max_tokens=500):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(content)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(encoding.decode(chunk))
    return chunks

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        content = ""
        for page in pdf:
            content += page.get_text()
        return content
    return ""

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    content = extract_text_from_file(uploaded_file)
    st.text_area("📄 Preview", content[:1000], height=300)

    chunks = split_into_chunks(content)

    embedding_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=chunks
    )
    embeddings = [e.embedding for e in embedding_response.data]

    user_question = st.text_input("💬 Ask a question about the content")
    if st.button("Ask GPT") and user_question:
        question_embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=user_question
        )
        question_embedding = question_embedding_response.data[0].embedding

        scored_chunks = []
        for chunk, emb in zip(chunks, embeddings):
            score = cosine_similarity(question_embedding, emb)
            scored_chunks.append((score, chunk))

        scored_chunks.sort(reverse=True)
        top_chunks = [chunk for _, chunk in scored_chunks[:3]]
        relevant_text = "\n\n".join(top_chunks)

        prompt = f"""You are a helpful assistant. Use the following document excerpts to answer the question.

Document:
{relevant_text}

Question:
{user_question}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        st.markdown("### 💡 GPT's Answer")
        st.write(response.choices[0].message.content)