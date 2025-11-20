from flask import Flask, render_template, request, jsonify
from supabase import create_client
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ---------------------------
# 1. Supabase connection
# ---------------------------
SUPABASE_URL = "https://lgcvwuazwdkrchxhvbcv.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxnY3Z3dWF6d2RrcmNoeGh2YmN2Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzM3MTgxOSwiZXhwIjoyMDc4OTQ3ODE5fQ.O6MPj2mePdwLb-G52ijDMgnIdFMU0NR41p6re0rRCAs"  # <-- Replace this soon, DO NOT expose publicly
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# 2. Gemini API setup
# ---------------------------
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Gemini text model
gemini_model = genai.GenerativeModel('models/gemini-2.5-pro')

print("Application initialized successfully.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    data = request.json
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        # ------------------------------------
        # STEP 1 — Get Gemini embedding (light, fast)
        # ------------------------------------
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=user_query,
            task_type="retrieval_query"
        )
        query_embedding = result['embedding']

        # ------------------------------------
        # STEP 2 — Query Supabase Vector DB
        # ------------------------------------
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": 20
            }
        ).execute()

        docs = response.data or []

        # Format results
        results = []
        chunks_text = ""
        for i, row in enumerate(docs, 1):
            results.append({
                "index": i,
                "content": row["content"],
                "similarity": row.get("similarity", "N/A")
            })
            chunks_text += f"\n\n--- Chunk {i} ---\n{row['content']}"

        # ------------------------------------
        # STEP 3 — Ask Gemini to refine the answer
        # ------------------------------------
        prompt = f"""
User Question: {user_query}

Context:
{chunks_text}

Instructions:
- Answer ONLY using the context.
- Use markdown formatting.
- Include LaTeX math when needed.
- Do NOT reference chunk numbers.
"""

        gemini_response = gemini_model.generate_content(prompt)
        refined_answer = gemini_response.text

        return jsonify({
            "results": results,
            "count": len(results),
            "refined_answer": refined_answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
