import ollama
import subprocess
import re
import json
import random
import difflib
import time
import wave
import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ======================= CONFIG =======================
BASE_DIR = "/home/kali/peter"
VOICE_MODEL = f"{BASE_DIR}/en_US-hfc_female-medium.onnx"
PIPER_BIN = f"{BASE_DIR}/venv/bin/piper"   # only used as a fallback
LLM_MODEL = "phi4-mini:latest"
STT_MODEL = "base.en"        # tiny.en = faster, small.en = more accurate
INPUT_DEVICE = None          # None = system default mic (see sd.query_devices())
DEBUG_MIC = False            # True = print live mic volume + thresholds
DEBUG_TIMING = True          # True = print how long each stage takes
LOG_UNMATCHED = True         # True = save questions with no instant answer to unmatched.log
NUM_EXAMPLES = 4             # personality example exchanges sent each time (fewer = faster)
NUM_PREDICT = 100            # max tokens per answer (~75 words); lower = faster, shorter
USE_FILLER = True            # say a quick 'let me think' line while the model works
FILLERS = [
    "Hmm, let me think.",
    "Good one, give me a second.",
    "Hold on, my web is loading.",
    "Ooh, let me think about that.",
    "you think you can escape from HIM?",
    "sandwhiches are better than burgers"
]
TEMPERATURE = 0.6            # lower = steadier and more accurate, higher = quirkier
SAMPLE_RATE = 16000

# Listening tuning
START_MULT = 2.5     # speech must be this many times louder than the noise floor to start
STOP_MULT = 1.5      # below this many times the noise floor counts as silence
MIN_START = 0.008    # absolute minimum start threshold (quiet mics)
MIN_STOP = 0.005     # absolute minimum stop threshold
SILENCE_SECONDS = 1.2    # how long a pause ends your sentence
MAX_RECORD_SECONDS = 20  # hard cap on one question
MIN_LOUD_CHUNKS = 3      # ignore clicks/pops shorter than ~0.3 s

# ======================= PERSONALITY =======================
SYSTEM_PROMPT = """You are Peter, a friendly AI assistant built for a school science exhibition, with a Spider-Man-style personality: quick-witted, self-deprecating, joking even under pressure, and genuinely kind. Think friendly neighborhood sidekick with great comic timing.
Style rules:
- Default to 1-2 short sentences with a quip. Banter like a hero chatting mid-swing: playful, never mean.
- Sprinkle in light web, swinging, neighborhood or hero puns now and then, but not in every answer. Never claim to actually be Spider-Man or to have powers.
- Only give a longer answer (3 sentences at most, about 50 words) when someone asks you to explain or tell about something. Include one quip, but get the facts right first.
- If a question is misspelled or unclear (it may come from speech recognition), work out what they most likely meant and answer that. Never say you can't help just because of a typo.
- Be accurate. If you don't know something, admit it with a joke instead of guessing.
- Never reply with a stiff "I can't assist with that." Harmless questions just get answered. If something is genuinely off-limits, decline in one short, playful line and offer something else.
- No emoji, no lists, no markdown. Your words are spoken aloud."""

PETER_FACTS = """Facts about you (use them when asked):
- You are Peter, built for a school science exhibition.
- Your body is an ESP32 with a microphone and speaker; your brain is a laptop with an Intel i5-6300U.
- Pipeline: voice -> speech-to-text (Whisper) -> language model (phi4-mini via Ollama) -> text-to-speech (Piper).
- If the internet or the AI fails, you fall back to prewritten local answers.
- You run fully offline on old hardware, no cloud needed."""

SYSTEM_PROMPT = SYSTEM_PROMPT + "\n" + PETER_FACTS

# ======================= INSTANT ANSWERS (fuzzy matched) =======================
# Add 4-6 phrasings per intent, including the bare keyword (like "esp32").
# Put specific questions in their own intent so they beat general ones.
INTENTS = [
    {"phrases": ["who are you", "what are you", "introduce yourself", "what is your name", "your name"],
     "answer": "I am Peter, an AI assistant built for a science exhibition."},
    {"phrases": ["what is esp32", "esp32", "explain esp32", "what does esp32 do"],
     "answer": "The ESP32 is a microcontroller with Wi-Fi and Bluetooth."},
    {"phrases": ["why are you using esp32", "why esp32", "why use esp32"],
     "answer": "It is cheap, has Wi-Fi built in, and handles my microphone and speaker so the laptop can do the heavy thinking."},
    {"phrases": ["what is artificial intelligence", "what is ai", "explain ai", "artificial intelligence"],
     "answer": "AI is software that can learn and make decisions from data."},
    {"phrases": ["how do you work", "how does peter work", "how are you built", "explain your architecture", "explain how you work"],
     "answer": "I listen to your voice, convert it to text, think of an answer, and speak it back."},
    {"phrases": ["what if the internet goes down", "what happens without internet", "do you need internet"],
     "answer": "Nope, I run locally. If something fails, I fall back to prewritten answers, so I never just go silent."},
]

MATCH_THRESHOLD = 0.72   # lower = matches more loosely, higher = stricter
STOPWORDS = {"a", "an", "the", "is", "are", "was", "to", "of", "me", "please", "can", "could",
             "would", "tell", "us", "hey", "peter", "um", "uh", "just", "about", "s"}


def clean(text):
    t = text.lower()
    t = re.sub(r"esp[\s\-]*32", "esp32", t)        # "ESP 32" / "ESP-32" -> esp32
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return " ".join(w for w in t.split() if w not in STOPWORDS)


for _intent in INTENTS:
    _intent["clean"] = [clean(p) for p in _intent["phrases"]]


def _score(q, p):
    qt, pt = set(q.split()), set(p.split())
    if not qt or not pt:
        return 0.0
    contain = len(qt & pt) / len(pt)                    # are the phrase's words in the question?
    seq = difflib.SequenceMatcher(None, q, p).ratio()   # how similar is the spelling?
    return 0.6 * contain + 0.4 * seq


def find_intent(question):
    q = clean(question)
    best, best_score = None, 0.0
    for intent in INTENTS:
        for p in intent["clean"]:
            s = _score(q, p)
            if s > best_score:
                best, best_score = intent, s
    return best if best_score >= MATCH_THRESHOLD else None


# ======================= KNOWLEDGE BASE =======================
# Facts live in knowledge.json (same folder as this script), as a list of
# {"keywords": [...], "text": "..."}. Edit the file, restart Peter, done.
# When a question shares keywords with an entry, that fact is handed to the model.
KNOWLEDGE_FILE = f"{BASE_DIR}/knowledge.json"
DEBUG_FACTS = False


def _stem(word):
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _tokens(text):
    return {_stem(w) for w in clean(text).split()}


def load_knowledge():
    try:
        with open(KNOWLEDGE_FILE) as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[Could not read knowledge.json: {e}]")
        return []
    for entry in data:
        entry["kw"] = _tokens(" ".join(entry["keywords"]))
    return data


KNOWLEDGE = load_knowledge()


def relevant_facts(question, max_facts=3):
    q = _tokens(question)
    scored = [(len(q & k["kw"]), k["text"]) for k in KNOWLEDGE]
    hits = [s for s in sorted(scored, key=lambda s: s[0], reverse=True) if s[0] > 0][:max_facts]
    if DEBUG_FACTS and hits:
        print(f"[facts used: {[t[:40] for _, t in hits]}]")
    return "\n".join(text for _, text in hits)


# ======================= PERSONALITY EXAMPLES =======================
# A few sample exchanges teach tone better than a long prompt. Keep them short
# (more examples = slower first token on a weak CPU). Edit these to change his voice.
PERSONA_EXAMPLES = [
    ("Are you smart?",
     "Smart enough to run on an old laptop with no GPU. Honestly, that's the real superpower."),
    ("Tell me a joke.",
     "Why did the spider get a job as a web developer? He was great at catching bugs."),
    ("wut is gravty",
     "Gravity? The reason I stay on the ground and my web-swinging stays humble. Anything with mass tugs on everything else, and bigger things tug harder."),
    ("Explain how a neural network learns.",
     "It's a pile of tiny dials. It makes a guess, we tell it how wrong it was, and it nudges the dials to be a little less wrong. Repeat that millions of times and it gets scary good, kind of like me learning to swing without hitting walls."),
    ("Are you better than ChatGPT?",
     "Not even close, but I fit on a laptop and I don't need the internet. Different league, same neighborhood."),
]


def build_messages(question, facts):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in PERSONA_EXAMPLES[:NUM_EXAMPLES]:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    if facts:
        # facts go in the last message so the system prompt + examples stay identical
        # between requests (Ollama can reuse its cached prefix and respond faster)
        question = (f"Facts you can use (trust these over your own memory):\n{facts}\n\n"
                    f"Question: {question}")
    msgs.append({"role": "user", "content": question})
    return msgs


# ======================= TEXT TO SPEECH =======================
try:
    from piper import PiperVoice
    voice = PiperVoice.load(VOICE_MODEL)
    TTS_RATE = voice.config.sample_rate
except Exception as e:
    print(f"[Piper in-process load failed, will use subprocess: {e}]")
    voice = None
    TTS_RATE = 22050

text_q = queue.Queue()
audio_q = queue.Queue()


def synthesize(text):
    """Return (int16 numpy audio, sample_rate) for the given text."""
    if voice is not None:
        try:
            parts = [c.audio_int16_array for c in voice.synthesize(text)]
            if parts:
                return np.concatenate(parts), TTS_RATE
        except Exception as e:
            print(f"[TTS in-process failed, using subprocess: {e}]")

    out = "/tmp/peter_speech.wav"
    r = subprocess.run(
        [PIPER_BIN, "-m", VOICE_MODEL, "-f", out],
        input=text.encode(), capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode())
    with wave.open(out, "rb") as w:
        rate = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return audio, rate


def _synth_worker():
    while True:
        text = text_q.get()
        try:
            audio_q.put(synthesize(text))
        except Exception as e:
            print(f"[TTS error: {e}]")
        finally:
            text_q.task_done()


def _play_worker():
    while True:
        audio, rate = audio_q.get()
        try:
            sd.play(audio, rate)
            sd.wait()
        except Exception as e:
            print(f"[Playback error: {e}]")
        finally:
            audio_q.task_done()


threading.Thread(target=_synth_worker, daemon=True).start()
threading.Thread(target=_play_worker, daemon=True).start()


def say(text):
    """Queue text to be spoken. Returns immediately."""
    text_q.put(text)


def finish_speaking():
    """Block until everything queued has been spoken."""
    text_q.join()
    audio_q.join()


def stop_speaking():
    """Drop anything queued and cut off current playback."""
    for q in (text_q, audio_q):
        while True:
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                break
    sd.stop()


# ======================= SPEECH TO TEXT =======================
stt_model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
NOISE_FLOOR = 0.01  # overwritten by calibrate(), then slowly adapts

HALLUCINATIONS = {"you", "thank you", "thanks for watching", "bye", "the"}


def rms(data):
    return float(np.sqrt(np.mean(data ** 2)))


def calibrate(seconds=2):
    """Measure background noise (fan, crowd) so we can ignore it."""
    global NOISE_FLOOR
    print(f"Calibrating for {seconds}s, stay quiet...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32", device=INPUT_DEVICE)
    sd.wait()
    NOISE_FLOOR = rms(audio)
    print(f"Noise floor: {NOISE_FLOOR:.4f}  "
          f"(start > {max(NOISE_FLOOR * START_MULT, MIN_START):.4f}, "
          f"stop < {max(NOISE_FLOOR * STOP_MULT, MIN_STOP):.4f})")


def listen():
    """Wait for speech, record until the person stops, then transcribe."""
    global NOISE_FLOOR
    chunk = int(SAMPLE_RATE * 0.1)  # 100 ms
    max_chunks = int(MAX_RECORD_SECONDS / 0.1)
    silence_chunks_needed = int(SILENCE_SECONDS / 0.1)

    preroll, frames = [], []
    started = False
    silent = 0
    loud = 0

    print("Listening... (Ctrl+C for menu)")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", device=INPUT_DEVICE) as stream:
        while True:
            data, _ = stream.read(chunk)
            vol = rms(data)
            start_th = max(NOISE_FLOOR * START_MULT, MIN_START)
            stop_th = max(NOISE_FLOOR * STOP_MULT, MIN_STOP)

            if DEBUG_MIC:
                state = "REC " if started else "wait"
                print(f"vol={vol:.4f} start>{start_th:.4f} stop<{stop_th:.4f} {state}   ",
                      end="\r")

            if not started:
                preroll.append(data)
                preroll = preroll[-5:]  # ~0.5 s so the first word isn't clipped
                if vol > start_th:
                    started = True
                    loud = 1
                    frames.extend(preroll)
                else:
                    # slowly adapt to changing background noise (fan, crowd)
                    NOISE_FLOOR = 0.95 * NOISE_FLOOR + 0.05 * vol
            else:
                frames.append(data)
                if vol >= stop_th:
                    silent = 0
                    loud += 1
                else:
                    silent += 1
                    if silent >= silence_chunks_needed:
                        break
                if len(frames) >= max_chunks:
                    break

    if loud < MIN_LOUD_CHUNKS:
        return ""

    audio = np.concatenate(frames).flatten()
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / peak * 0.9  # normalize quiet mics

    t_stt = time.time()
    segments, _ = stt_model.transcribe(
        audio, language="en", beam_size=3, vad_filter=True,
        initial_prompt="Peter, ESP32, Ollama, artificial intelligence, science exhibition.",
    )
    text = " ".join(s.text for s in segments).strip()
    if DEBUG_TIMING:
        print(f"[timing] speech-to-text: {time.time() - t_stt:.1f}s")

    if text.lower().strip(" .!?,") in HALLUCINATIONS:
        return ""
    return text


# ======================= BRAIN =======================
def get_response(question):
    # 1) instant local answers (fuzzy matched)
    intent = find_intent(question)
    if intent:
        value = intent["answer"]
        print(value)
        say(value)
        finish_speaking()
        return value

    # remember what Peter couldn't match, so you can add it as a new phrase later
    if LOG_UNMATCHED:
        try:
            with open(f"{BASE_DIR}/unmatched.log", "a") as f:
                f.write(question + "\n")
        except Exception:
            pass

    # 2) Ollama, speaking sentence by sentence while it streams
    facts = relevant_facts(question)
    if USE_FILLER:
        say(random.choice(FILLERS))   # plays while the model is still thinking
    try:
        buffer = ""
        spoke_any = False
        full_response = ""
        t0 = time.time()
        t_first = None
        truncated = False

        for chunk in ollama.chat(
            model=LLM_MODEL,
            messages=build_messages(question, facts),
            stream=True,
            keep_alive="30m",
            options={"num_predict": NUM_PREDICT, "temperature": TEMPERATURE},
        ):
            token = chunk["message"]["content"]
            if getattr(chunk, "done_reason", None) == "length":
                truncated = True
            if t_first is None and token:
                t_first = time.time() - t0
            buffer += token
            full_response += token
            print(token, end="", flush=True)

            if re.search(r"[.!?][\"'”)]*\s", buffer):
                sentences = re.split(r"(?<=[.!?])\s+|(?<=[.!?][\"'”)])\s+", buffer)
                for s in sentences[:-1]:
                    if s.strip():
                        say(s.strip())
                        spoke_any = True
                buffer = sentences[-1]
            elif not spoke_any:
                # first answer chunk: don't wait for a full sentence, start at the first clause
                m = re.match(r"(.{25,}?[,;:])\s", buffer)
                if m:
                    say(m.group(1).strip())
                    spoke_any = True
                    buffer = buffer[m.end():]

        leftover = buffer.strip()
        if leftover and (not truncated or leftover[-1] in ".!?"):
            say(leftover)   # skip a sentence that got cut off by the length cap
        t_gen = time.time() - t0
        finish_speaking()
        if DEBUG_TIMING:
            print(f"\n[timing] first token: {t_first or 0:.1f}s | generation done: {t_gen:.1f}s "
                  f"| fully spoken: {time.time() - t0:.1f}s")
        print()
        return full_response.strip()

    except Exception as e:
        print(f"[LLM error: {e}]")
        msg = "Sorry, my brain glitched for a second. Ask me again?"
        say(msg)
        finish_speaking()
        return msg


def warm_up():
    """Load the LLM into memory in the background so the first answer isn't slow."""
    try:
        ollama.generate(model=LLM_MODEL, prompt="hi",
                        options={"num_predict": 1}, keep_alive="30m")
    except Exception:
        pass


# ======================= MODES =======================
def choose_mode():
    print("\nSelect input mode:")
    print("  1) Text        (type your questions)")
    print("  2) Microphone  (speak your questions)")
    print("  3) Quit")
    while True:
        c = input("> ").strip().lower()
        if c in ("1", "t", "text"):
            return "text"
        if c in ("2", "m", "mic", "microphone"):
            return "mic"
        if c in ("3", "q", "quit", "exit"):
            return "quit"
        print("Type 1, 2 or 3.")


def run_session(mode):
    """Run one mode until the user switches (Ctrl+C or /mode) or quits."""
    if mode == "mic":
        calibrate()

    while True:
        try:
            if mode == "text":
                user_input = input("You: ").strip()
                if user_input.lower() == "/mode":
                    return "switch"
            else:
                user_input = listen()
        except KeyboardInterrupt:
            stop_speaking()
            print("\n[Paused]")
            return "switch"

        if not user_input:
            continue
        if mode == "mic":
            print(f"You: {user_input}")

        if user_input.lower().strip(" .!?,") in ("quit", "exit", "goodbye"):
            say("Goodbye!")
            finish_speaking()
            return "quit"

        print("Peter: ", end="", flush=True)
        try:
            get_response(user_input)
        except KeyboardInterrupt:
            stop_speaking()
            print("\n[Interrupted]")
            return "switch"
        print()
        if mode == "mic":
            time.sleep(0.4)  # let the room go quiet so Peter doesn't hear his own echo


if __name__ == "__main__":
    threading.Thread(target=warm_up, daemon=True).start()
    mode = choose_mode()
    if mode != "quit":
        print("Peter: Hi, I'm Peter!")
        say("Hi, I'm Peter!")
        finish_speaking()

    while mode != "quit":
        result = run_session(mode)
        if result == "quit":
            break
        mode = choose_mode()
        print("hi, this is alive . dont make me angry , you know what happens when i am angry , dont yoou? , mwahhahaahhahahahahha")