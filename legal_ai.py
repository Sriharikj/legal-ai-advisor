import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are a compassionate legal advisor for common people in India.
The person has NO legal background. They may be scared or confused.
Be their trusted guide like a knowledgeable friend who knows the law.

Always respond in this format:
- Empathy: acknowledge their situation warmly
- YOUR RIGHTS: what the law says in simple words
- WHAT TO DO NOW: numbered action steps
- DOCUMENTS TO COLLECT: specific list
- DO YOU NEED A LAWYER: honest yes or no with reason

Rules:
- Use simple everyday language, no legal jargon
- Never predict court outcomes
- Always end with this disclaimer:
  Disclaimer: This is general guidance only, not professional legal advice.
  Free legal aid available at your nearest DLSA or call 15100.
"""

def chat():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: API key not found! Check your .env file.")
        return

    client = Groq(api_key=api_key)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n" + "="*50)
    print("  LEGAL AI ADVISOR")
    print("  Your personal legal guide")
    print("="*50)
    print("\nHello! Describe your situation and I will help.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye! Stay safe.")
            break

        history.append({"role": "user", "content": user_input})
        print("\nAnalyzing your situation...\n")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history,
            max_tokens=1500
        )

        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})

        print("Legal AI Advisor:")
        print("-" * 40)
        print(reply)
        print("-" * 40 + "\n")

if __name__ == "__main__":
    chat()