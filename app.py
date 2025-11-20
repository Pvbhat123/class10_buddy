from flask import Flask, render_template, request, jsonify
from supabase import create_client
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize embedding model with cache and optimization
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 1}
)

# Read from environment variables for security
SUPABASE_URL = os.getenv('SUPABASE_URL', "https://lgcvwuazwdkrchxhvbcv.supabase.co")
SUPABASE_KEY = os.getenv('SUPABASE_KEY', "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxnY3Z3dWF6d2RrcmNoeGh2YmN2Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzM3MTgxOSwiZXhwIjoyMDc4OTQ3ODE5fQ.O6MPj2mePdwLb-G52ijDMgnIdFMU0NR41p6re0rRCAs")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure Gemini - Read from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found!")
    print("Please create a .env file with your API key:")
    print("GEMINI_API_KEY=your_actual_api_key_here")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('models/gemini-2.5-pro')

print("Connected successfully!")
print(f"Using Gemini model: gemini-2.5-flash")
print(f"API key: {GEMINI_API_KEY[:10]}...")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    user_query = data.get('query', '').strip()

    if not user_query:
        return jsonify({'error': 'Query cannot be empty'}), 400

    try:
        # Embed query
        query_embedding = embeddings.embed_query(user_query)

        # Call Supabase RPC
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": 20
            }
        ).execute()

        docs = response.data

        if not docs:
            return jsonify({
                'results': [],
                'count': 0,
                'refined_answer': 'No relevant information found to answer your question.'
            })

        # Format results
        results = []
        chunks_text = ""
        for i, row in enumerate(docs, 1):
            results.append({
                'index': i,
                'content': row["content"],
                'similarity': row.get("similarity", "N/A")
            })
            chunks_text += f"\n\n--- Chunk {i} ---\n{row['content']}"

        # Create prompt for Gemini
        prompt = f"""You are an intelligent assistant that answers questions based on provided context.

User Question: {user_query}

Context Information (Retrieved Chunks):
{chunks_text}

Instructions:
1. Carefully analyze the context provided above
2. Answer the user's question accurately based ONLY on the information in the context
3. Format your answer in a clear, well-structured manner with proper sections if needed
4. Use markdown formatting for better readability (headings, bullet points, bold, etc.)
5. For mathematical formulas and equations:
   - Use LaTeX notation wrapped in $ for inline math: $m = h'/h$
   - Use $$ for block/display math equations
   - Example: "The magnification formula is $m = \\frac{{h'}}{{h}} = \\frac{{v}}{{u}}$"
   - Example: "The mirror equation is $$\\frac{{1}}{{v}} - \\frac{{1}}{{u}} = \\frac{{1}}{{f}}$$"
   - Always explain what each variable represents after the formula
6. If the context doesn't contain enough information to fully answer the question, mention this clearly
7. Be concise but comprehensive
8. DO NOT mention chunk numbers or cite chunks (like "Chunk 1", "Chunk 5", etc.) in your answer
9. Present the information naturally without referencing the source chunks

Please provide a detailed, well-formatted answer WITHOUT any chunk references:"""

        # Send to Gemini
        gemini_response = gemini_model.generate_content(prompt)
        refined_answer = gemini_response.text

        return jsonify({
            'results': results,
            'count': len(results),
            'refined_answer': refined_answer
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Use PORT from environment for deployment, default to 5000 for local
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
