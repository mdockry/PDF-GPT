# 📄🤖 PDF-GPT

**PDF-GPT** is an AI-powered assistant that lets you upload a PDF, ask questions about it.

Built with **Streamlit**, **OpenAI**, and **tiktoken**.

---

## 🚀 Features

- 🧠 Ask questions about your PDF content
- 📚 Summarizes long documents intelligently
- 🧩 Handles long files via chunking + embedding
- 🧾 Simple interface built with Streamlit

---

## 📸 Preview

<table>
  <tr>
    <td><img src="textassistant.png" width="400"/></td>
    <td><img src="textassistant2.png" width="400"/></td>
  </tr>
</table>

---

## 📦 Requirements

```bash
pip install -r requirements.txt
``` 

You’ll also need an `.env` file containing your OpenAI key:

```ini
OPENAI_API_KEY=your-key-here
```

## 🛠️ Tech Stack

- Streamlit  
- OpenAI API  
- tiktoken  
- PyMuPDF (`fitz`)  
- NumPy  

## ✅ How it Works

1. Upload a PDF  
2. It’s split into ~500-token chunks  
3. Each chunk is embedded using `text-embedding-ada-002`  
4. Your question is also embedded  
5. Cosine similarity finds the most relevant chunks  
6. GPT generates a response based on those chunks  

