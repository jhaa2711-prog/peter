import ollama

# Peter's personality
SYSTEM_PROMPT = """You are Peter, an AI assistant . 
Be genuine and conversational, not forced or fake.
Use humor naturally when it fits, but don't force it.
Be knowledgeable and precise. Keep answers short (5-6 sentences max).dont be boring . dont give intentionall long sentences , only give long answers the prompt
is to explain or tell abbout something , in other times just give short humarous answers
""" 
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
    
    # Check fallback 
    for key, value in FALLBACK.items():
        if key in question_lower:
            return value
    
    # Ask Ollama streaming
    try:
        full_response = ""
        for chunk in ollama.generate(
            model="phi4-mini:latest",
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