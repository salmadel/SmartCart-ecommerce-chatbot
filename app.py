import os
import io
import base64
import pandas as pd
import numpy as np
from PIL import Image
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, CrossEncoder
from google.cloud import aiplatform_v1
from google.cloud import speech_v1p1beta1 as speech
from vertexai.preview.generative_models import GenerativeModel, Tool, FunctionDeclaration
from langchain_community.chat_message_histories import FirestoreChatMessageHistory

# -------------- 1. Load Data ----------------

df = pd.read_csv("products.csv")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

# -------------- 2. Load Models ----------------

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
image_embed_model = SentenceTransformer("clip-ViT-B-32")


# -------------- 3. Vertex AI Vector Search Setup ----------------

API_ENDPOINT = "316910219.us-central1-664415384988.vdb.vertexai.goog"
INDEX_ENDPOINT = "projects/664415384988/locations/us-central1/indexEndpoints/3767368840434941952"
DEPLOYED_INDEX_ID = "products_vector_index_1753792314645"
DEPLOYED_IMAGES_INDEX_ID = "products_images_vector_ind_1753792254396"

client_options = {"api_endpoint": API_ENDPOINT}
vector_search_client = aiplatform_v1.MatchServiceClient(
    client_options=client_options)


# -------------- 4. Gemini Model & Tools ----------------

search_products_fn = FunctionDeclaration(
    name="search_products",
    description=(
        "Use this function ONLY when the user asks about product information "
        "(price, features, description, rating, etc.). "
        "The input is a search query string."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Product search query text, e.g. 'LOVEVOOK Diaper Bag Backpack'"
            }
        },
        "required": ["query"],
    },
)

search_products_by_image_fn = FunctionDeclaration(
    name="search_products_by_image",
    description=(
        "Use this function ONLY when the user uploads a product image "
        "and wants to find matching products by image similarity."
    ),
    parameters={
        "type": "object",
        "properties": {
            "image_base64": {
                "type": "string",
                "description": "Base64 encoded image string sent by user."
            }
        },
        "required": ["image_base64"],
    },
)

tools = [
    Tool(function_declarations=[
         search_products_fn, search_products_by_image_fn])
]

model = GenerativeModel("gemini-2.5-flash", tools=tools)


# -------------- 5. Firestore Chat History ----------------

chat_history = FirestoreChatMessageHistory(
    collection_name="chat_sessions",
    session_id="rag_single_session",
    user_id="salma_01"
)


# -------------- 6. Helper Functions ----------------

def vector_search(query_text=None, image_embedding=None, df=None, mode="text"):
    if mode == "text":
        query_vector = embed_model.encode(query_text).tolist()
    elif mode == "image":
        query_vector = image_embedding
    else:
        raise ValueError("Invalid mode. Use 'text' or 'image'.")

    datapoint = aiplatform_v1.IndexDatapoint(feature_vector=query_vector)
    query = aiplatform_v1.FindNeighborsRequest.Query(
        datapoint=datapoint, neighbor_count=10
    )

    if mode == "text":
        deployed_index_id = DEPLOYED_INDEX_ID
    elif mode == "image":
        deployed_index_id = DEPLOYED_IMAGES_INDEX_ID
    else:
        raise ValueError("Invalid mode. Use 'text' or 'image'.")

    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=INDEX_ENDPOINT,
        deployed_index_id=deployed_index_id,
        queries=[query],
        return_full_datapoint=True
    )
    vs_response = vector_search_client.find_neighbors(request=request)
    neighbor_ids = [
        n.datapoint.datapoint_id for n in vs_response.nearest_neighbors[0].neighbors]
    candidates = df[df['asin'].isin(neighbor_ids)]
    return candidates, neighbor_ids


def rerank(query_text, docs):
    pairs = [(query_text, row['full_text']) for _, row in docs.iterrows()]
    scores = cross_encoder.predict(pairs)
    docs = docs.copy()
    docs['score'] = scores
    docs = docs.sort_values(by='score', ascending=False)
    return docs


def get_image_embedding_from_base64(image_base64_str):
    image_bytes = base64.b64decode(image_base64_str)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    embedding = image_embed_model.encode(image)
    return embedding.tolist()


def convert_role(msg_type):
    if msg_type == "human":
        return "user"
    elif msg_type == "ai":
        return "model"
    else:
        return "system"


def speech_to_text(audio_bytes):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code="en-US",
        enable_automatic_punctuation=True
    )
    response = client.recognize(config=config, audio=audio)
    transcripts = [
        result.alternatives[0].transcript for result in response.results]
    return " ".join(transcripts)


# -------------- 7. RAG Pipeline Function ----------------

def rag_pipeline(user_query=None, user_image_base64=None, user_audio_bytes=None):
    if df is None:
        raise ValueError("DataFrame df must be provided.")

    if user_audio_bytes:
        user_query = speech_to_text(user_audio_bytes)
        print(f"[Voice Recognized]: {user_query}")

    if user_query:
        chat_history.add_user_message(user_query)

    messages = [
        {"role": convert_role(msg.type), "parts": [{"text": msg.content}]}
        for msg in chat_history.messages
    ]

    if user_image_base64:
        image_embedding = get_image_embedding_from_base64(user_image_base64)
        candidates, _ = vector_search(
            image_embedding=image_embedding, df=df, mode="image")
        if candidates.empty:
            reply = "Sorry, no matching products found for the image."
            chat_history.add_ai_message(reply)
            return {"answer": reply}

        ranked = rerank(user_query or "", candidates)
        best_doc = ranked.iloc[0]

        final_prompt = f"""You are an expert shopping assistant.
User question: "I sent a product image to find matching products."
Below is detailed product information:
{best_doc['full_text']}
Product URL: {best_doc['productURL']}
Answer using ONLY the above info."""

        final_messages = messages + \
            [{"role": "user", "parts": [{"text": final_prompt}]}]

        final_response = model.generate_content(contents=final_messages)
        part = final_response.candidates[0].content.parts[0]

        if hasattr(part, "function_call") and part.function_call:
            print("[LOG] Unexpected function_call in image flow!")
            fallback = "Sorry, couldn't finalize the answer for the image. Please try again."
            chat_history.add_ai_message(fallback)
            return {"answer": fallback, "sources": [best_doc['asin']]}

        final_text = part.text.strip()
        chat_history.add_ai_message(final_text)
        return {"answer": final_text, "transcript": None, "sources": [best_doc['asin']]}

    response = model.generate_content(contents=messages)
    part = response.candidates[0].content.parts[0]

    if hasattr(part, "function_call") and part.function_call:
        fn_name = part.function_call.name
        fn_args = part.function_call.args
        print(f"[LOG] Function call detected: {fn_name} with args: {fn_args}")
        if fn_name == "search_products":
            query_text = fn_args["query"]
            candidates, _ = vector_search(
                query_text=query_text, df=df, mode="text")
            if candidates.empty:
                reply = "Sorry, no matching products found."
                chat_history.add_ai_message(reply)
                return {"answer": reply}

            ranked = rerank(query_text, candidates)
            best_doc = ranked.iloc[0]

            final_prompt = f"""You are an expert shopping assistant.
User question: "{user_query}"
Below is product info:
{best_doc['full_text']}
Product URL: {best_doc['productURL']}
Answer clearly using ONLY this info."""

            final_messages = messages + \
                [{"role": "user", "parts": [{"text": final_prompt}]}]

            final_response = model.generate_content(contents=final_messages)
            part = final_response.candidates[0].content.parts[0]

            if hasattr(part, "function_call") and part.function_call:
                print("[LOG] Unexpected function_call in re-ranking answer step!")
                fallback = "Sorry, couldn't finalize the answer. Please try again."
                chat_history.add_ai_message(fallback)
                return {"answer": fallback, "transcript": user_query, "sources": [best_doc['asin']]}

            final_text = part.text.strip()
            chat_history.add_ai_message(final_text)
            return {"answer": final_text, "transcript": user_query, "sources": [best_doc['asin']]}

    if hasattr(part, "function_call") and part.function_call:
        print(f"[LOG] Unexpected function_call fallback: {part.function_call}")
        fallback = "Sorry, I couldn't find matching results."
        chat_history.add_ai_message(fallback)
        return {"answer": fallback, "transcript": user_query}

    reply = part.text.strip()
    chat_history.add_ai_message(reply)
    return {"answer": reply, "transcript": user_query}


# -------------- 8. Flask App Setup ----------------

app = Flask(__name__, static_url_path="/static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def handle_query():
    data = request.json
    user_query = data.get("query")
    user_image_base64 = data.get("image_base64")
    user_audio_base64 = data.get("audio_base64")

    user_audio_bytes = None
    if user_audio_base64:
        user_audio_bytes = base64.b64decode(user_audio_base64)

    try:
        result = rag_pipeline(
            user_query=user_query,
            user_image_base64=user_image_base64,
            user_audio_bytes=user_audio_bytes
        )
        print("[DEBUG] API Response:", result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
