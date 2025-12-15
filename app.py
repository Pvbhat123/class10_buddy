from flask import Flask, render_template, request, jsonify
from supabase import create_client
import google.generativeai as genai
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------
# 1. HUGGINGFACE INFERENCE API FOR EMBEDDINGS
# ---------------------------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"

def get_embedding(text):
    """Get embeddings using HuggingFace Inference API (free)"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"HuggingFace API error: {response.text}")

# ---------------------------------------------
# 2. SUPABASE CONFIG
# ---------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------
# 3. GEMINI CONFIG
# ---------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-robotics-er-1.5-preview")


@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------
# 🔥 HELPER FUNCTION — Fetch image based on query keywords
# ---------------------------------------------------------
def fetch_image_for_query(query):
    query = query.lower()

    # Keyword detection logic
    keywords = ["diagram", "image", "picture", "labelled", "structure"]

    if any(k in query for k in keywords):
        # Search in Supabase table using ILIKE (case-insensitive)
        response = supabase.table("textbook_images").select("*") \
            .ilike("topic", f"%{query}%").execute()

        if response.data:
            return response.data[0]["image_url"]

    return None


# ---------------------------------------------------------
#  SEARCH ROUTE WITH IMAGE SUPPORT
# ---------------------------------------------------------
@app.route("/search", methods=["POST"])
def search():
    data = request.json
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        # 🔍 Check if user is asking for a diagram
        image_url = fetch_image_for_query(user_query)

        # If user ONLY wants an image → skip RAG, return image immediately
        if image_url and len(user_query.split()) <= 4:
            return jsonify({
                "results": [],
                "count": 0,
                "refined_answer": "Here is the diagram you requested:",
                "image_url": image_url
            })

        # 1. Embed query using HuggingFace API
        query_embedding = get_embedding(user_query)

        # 2. Supabase Vector Search
        response = supabase.rpc(
            "match_documents",
            {"query_embedding": query_embedding, "match_count": 20}
        ).execute()

        docs = response.data or []

        if not docs:
            return jsonify({
                "results": [],
                "count": 0,
                "refined_answer": "No relevant information found.",
                "image_url": image_url
            })

        # 3. Prepare chunks
        results = []
        context_text = ""

        for i, row in enumerate(docs, 1):
            results.append({
                "index": i,
                "content": row["content"],
                "similarity": float(row.get("similarity", 0))
            })
            context_text += f"\n\n--- CHUNK {i} ---\n{row['content']}"

        # 4. Ask Gemini
        prompt = f"""
Answer the question ONLY using the context provided.

USER QUESTION:
{user_query}

CONTEXT:
{context_text}

RULES:
- Do NOT mention chunks.
- Use only context.
- Use markdown formatting.
"""
        gemini_response = gemini_model.generate_content(prompt)
        refined_answer = gemini_response.text

        # Include image if found
        return jsonify({
            "results": results,
            "count": len(results),
            "refined_answer": refined_answer,
            "image_url": image_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
