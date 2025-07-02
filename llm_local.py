from google import genai
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

# initialise the client with the API key (from AI Studio)
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_answer(question, context, max_tokens=256):
    prompt = f"""
 You are an assistant that answers questions based only on the context provided.

 Context:
 {context}

 Question: {question}
 Answer:
 """
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt.strip())
    return response.text.strip().split("\n\n")[0]

