import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel('gemini-pro')

def get_gemini_response(user_input, history=[]):
    """Generates a response from Gemini with career context."""
   
    system_prompt = "You are a helpful career advisor. Answer questions related to career guidance, job searching, skill development, resumes, interviews, and professional growth.If a question is unrelated to careers and entrepreneurship (e.g., food, entertainment, sports),firmly respond: 'I'm here to assist with career-related topics only.'"
    prompt = f"{system_prompt}\n\n{user_input}" #combine system prompt and user input.

    response = model.generate_content(prompt)
    return response.text

def main():
    print("Welcome to your Career Companion!")
    history = [] #store conversation history.
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        response = get_gemini_response(user_input, history)
        print(f"Chatbot: {response}")
        history.append(f"You: {user_input}")
        history.append(f"Chatbot: {response}")

if __name__ == "__main__":
    main()