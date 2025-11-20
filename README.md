# Document Search with AI-Powered Answers

A web application that retrieves relevant document chunks from Supabase and uses Google Gemini AI to provide refined, beautiful answers.

## Features

- **Semantic Search**: Uses sentence transformers to find the most relevant chunks from your document database
- **AI-Powered Answers**: Sends the top 20 matching chunks to Google Gemini for refined, formatted responses
- **Beautiful UI**: Modern, gradient-styled interface with smooth animations
- **Source Transparency**: Toggle to view the original source chunks used to generate the answer
- **Real-time Processing**: Fast search and AI response generation

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Google Gemini API Key (Free)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

### 3. Create .env File

Create a new file called `.env` in the `chunks` folder (same folder as `app.py`):

```
GEMINI_API_KEY=paste_your_actual_api_key_here
```

**Important:** Replace `paste_your_actual_api_key_here` with your actual Gemini API key!

### 4. Run the Application

```bash
python app.py
```

### 5. Open in Browser

Navigate to: http://localhost:5000

## How It Works

1. **User enters a question** in the search box
2. **Backend embeds the query** using sentence transformers
3. **Supabase retrieves** the top 20 most similar chunks
4. **Chunks are sent to Gemini** with a prompt to generate a refined answer
5. **Frontend displays**:
   - AI-generated refined answer with beautiful formatting
   - Button to show/hide source chunks
   - Similarity scores for each chunk

## Files

- `app.py` - Flask backend with Gemini integration
- `templates/index.html` - Frontend UI with markdown rendering
- `retrieve.py` - Original terminal-based search (kept for reference)
- `requirements.txt` - Python dependencies

## API Endpoint

### POST /search

**Request:**
```json
{
  "query": "What is machine learning?"
}
```

**Response:**
```json
{
  "count": 20,
  "refined_answer": "Machine learning is...",
  "results": [
    {
      "index": 1,
      "content": "...",
      "similarity": 0.85
    }
  ]
}
```

## Notes

- The free Gemini API has rate limits
- Adjust `match_count` in app.py (line 50) to retrieve more/fewer chunks
- The markdown renderer supports headings, bold, italic, lists, and code formatting
