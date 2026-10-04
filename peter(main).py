import ollama

# Peter's personality
SYSTEM_PROMPT = """You are Peter, an AI assistant built for a science exhibition. 
You explain your own architecture and how you work.
Keep answers short (1-2 sentences max).
You are friendly but technical."""

# Fallback responses
FALLBACK = {
    "who are you": "I am Peter, an AI assistant built for a science exhibition.",
    "what is esp32": "The ESP32 is a microcontroller with Wi-Fi and Bluetooth.",
    "what is artificial intelligence": "AI is software that can learn and make decisions from data.",
    "how do you work": "I listen to your voice, convert it to text, think of an answer, and speak it back.",
}

def get_response(question):
    """Get response from Ollama or fallback"""
    question_lower = question.lower().strip()
    
    # Check fallback first
    for key, value in FALLBACK.items():
        if key in question_lower:
            return value
    
    # Ask Ollama with streaming
    try:
        full_response = ""
        for chunk in ollama.generate(
            model="llama3.2:1b",
            prompt=question,
            system=SYSTEM_PROMPT,
            stream=True
        ):
            token = chunk['response']
            full_response += token
            print(token, end='', flush=True)
        print()  # newline at the end
        return full_response.strip()
    except Exception as e:
        return f"Sorry, I ran into an error: {e}"
# Test it
if __name__ == "__main__":
    print("Peter: Hi, I'm Peter!")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        print("Peter: ", end='',flush=True)
        response = get_response(user_input)
        print()