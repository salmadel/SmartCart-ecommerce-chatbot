# 🛒 SmartCart — AI RAG Shopping Assistant

**SmartCart** is an intelligent shopping assistant that helps users find product information easily using **text**, **images**, or **voice**.  
The chatbot uses a **RAG (Retrieval-Augmented Generation)** pipeline combining **vector search**, **re-ranking**, and **Gemini LLM** for final answers.  
It also keeps a **chat history** for smoother user experience.



## Dataset

- SmartCart uses a sample of 20,000 products from the Amazon Products Dataset 2023 [(Kaggle link)](https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products).
- The sample includes product titles, descriptions, prices, ratings, ASINs, product page links, and image URLs.



## Features

- **Text Query** — Ask about any product details by text.
- **Image Query** — Upload an image to find similar products.
- **Voice Query** — Record your question and get it transcribed & answered.
- **RAG Pipeline** — Uses vector search + cross encoder re-ranking + Gemini.
- **Chat History** — Stores past messages in Firestore for context.



## Frontend Tools

- **HTML**: Structure of the chat window, input area, and buttons.
- **CSS**: Custom styling for message bubbles, voice recording bubble, and overall theme.
- **JavaScript**: Handles sending text, uploading images, recording audio, and displaying responses in real time.



## Backend Tools & Models

- **Flask**: Lightweight Python server.
- **SentenceTransformers**: `all-MiniLM-L6-v2` for text embeddings.
- **CLIP ViT-B-32**: For generating image embeddings.
- **Cross-Encoder**: `ms-marco-MiniLM-L-12-v2` for reranking candidates.
- **Vertex AI Vector Search**: Stores product and image embeddings.
- **Gemini 2.5 Flash**: Google LLM for generating final answers.
- **Google Cloud Speech-to-Text**: For converting voice recordings to text
- **LangChain Firestore History**: Stores the chat session history.



## How It Works (Pipeline)

1. User sends **text**, **image**, or **voice**.  
2. If voice, it’s transcribed using Google STT.
3. Text or image query is encoded → Vector Search (Vertex AI).  
4. Retrieved results are re-ranked with Cross Encoder.  
5. Top result is sent to Gemini to generate a final, context-aware answer.  
6. Answer + user question saved in Firestore.



## Test Video



https://github.com/user-attachments/assets/f6679ea9-ebb1-4a76-8d22-902106c5e1c8





