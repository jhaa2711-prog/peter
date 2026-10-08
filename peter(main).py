import ollama
import subprocess
import os
import re

# Peter's personality
SYSTEM_PROMPT = """You are Peter, an AI assistant built for a science exhibition. 
Be genuine and conversational, not forced or fake.
Use humor naturally when it fits, but don't force it.
Be knowledgeable and precise. Keep answers short (5-6 sentences max).
dont be boring . dont give intentionall long sentences , only give long answers when the prompt is to explain or tell about something , in other times just give short humorous answers
. dont use emoji . """

# Fallback responses
FALLBACK = {
    "who are you": "I am Peter, an AI assistant built for a science exhibition.",
    "what is esp32": "The ESP32 is a microcontroller with Wi-Fi and Bluetooth.",
    "what is artificial intelligence": "AI is software that can learn and make decisions from data.",
    "how do you work": "I listen to your voice, convert it to text, think of an answer, and speak it back.",
    "why are you":"i dont know , i was made to help people or something , and to just eat up electrons  on a random friday, but , lemme ask you something , 'why are, YOU ?'"
}
def speak(text):
    try:
        result = subprocess.run(
            ["/home/kali/peter/venv/bin/piper",
             "-m", "/home/kali/peter/en_US-hfc_female-medium.onnx",
             "-f", "/tmp/peter_speech.wav"],
            input=text.encode(),
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"Piper error: {result.stderr.decode()}")
            return
        subprocess.run(["aplay", "/tmp/peter_speech.wav"], stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"TTS Error: {e}")

def get_response(question):
    question_lower = question.lower().strip()

    for key, value in FALLBACK.items():
        if key in question_lower:
            print(value, end='', flush=True)
            speak(value)
            return value

    try:
        buffer = ""
        full_response = ""

        for chunk in ollama.generate(
            model="phi4-mini:latest",
            prompt=question,
            system=SYSTEM_PROMPT,
            stream=True
        ):
            token = chunk['response']
            buffer += token
            full_response += token
            print(token, end='', flush=True)

            # speak whenever we hit a sentence boundary
            if re.search(r'[.!?]\s', buffer):
                sentences = re.split(r'(?<=[.!?])\s+', buffer)
                for sentence in sentences[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        speak(sentence)
                buffer = sentences[-1]  # leftover fragment

        # speak whatever is left at the end
        if buffer.strip():
            speak(buffer.strip())

        print()
        return full_response.strip()

    except Exception as e:
        return f"Sorry, I ran into an error: {e}"

# Test it
if __name__ == "__main__":
    print("Peter: Hi, I'm Peter!")
    speak("Hi, I'm Peter!")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        print("Peter: ", end='', flush=True)
        response = get_response(question=user_input)
        print()