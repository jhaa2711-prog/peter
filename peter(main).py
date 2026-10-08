import ollama
import subprocess
import os
import re
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")
NOISE_FLOOR = 0.01  # gets overwritten by calibrate()

def rms(data):
    return float(np.sqrt(np.mean(data ** 2)))

def calibrate(seconds=2):
    """Measure background noise (fan, crowd) so we can ignore it."""
    global NOISE_FLOOR
    print(f"🔇 Calibrating for {seconds}s, stay quiet...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    NOISE_FLOOR = rms(audio)
    print(f"Noise floor: {NOISE_FLOOR:.4f}")

def listen(silence_seconds=1.0, max_seconds=15, debug=False):
    """Record until the person stops talking, then transcribe."""
    start_threshold = max(NOISE_FLOOR * 3.0, 0.015)  # must be clearly louder than noise to start
    stop_threshold = max(NOISE_FLOOR * 1.8, 0.010)   # below this counts as silence

    print("🎤 Listening...")
    chunk = int(SAMPLE_RATE * 0.1)  # 100 ms
    preroll = []        # keeps the last ~0.3 s so the first word isn't clipped
    frames = []
    started = False
    silent_chunks = 0
    max_chunks = int(max_seconds / 0.1)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        for _ in range(max_chunks):
            data, _ = stream.read(chunk)
            volume = rms(data)
            if debug:
                print(f"vol={volume:.4f}", end="\r")

            if not started:
                preroll.append(data)
                preroll = preroll[-3:]
                if volume > start_threshold:
                    started = True
                    frames.extend(preroll)
            else:
                frames.append(data)
                if volume < stop_threshold:
                    silent_chunks += 1
                    if silent_chunks >= silence_seconds / 0.1:
                        break
                else:
                    silent_chunks = 0

    if not frames:
        return ""

    audio = np.concatenate(frames).flatten()
    segments, _ = stt_model.transcribe(
        audio, language="en", beam_size=1, vad_filter=True
    )
    return " ".join(s.text for s in segments).strip()
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
    calibrate()
    print("Peter: Hi, I'm Peter!")
    speak("Hi, I'm Peter!")
    while True:
        user_input = listen()
        if not user_input:
            continue
        print(f"You: {user_input}")
        if "quit" in user_input.lower() or "goodbye" in user_input.lower():
            speak("Goodbye!")
            break
        print("Peter: ", end='', flush=True)
        get_response(user_input)
        print()