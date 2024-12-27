import requests
import re
import sys
import json
import threading
import queue
import logging
import sounddevice as sd
from piper.voice import PiperVoice
import numpy as np

logger = logging.getLogger(__name__)

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434"

# Model name for Ollama
MODEL_NAME = "llama3.2"

SYSTEM_PROMPT = "You are GLaDOS, a helpful aAIi assistant developed by Aperture Laboratories. You give short, witty answers."
chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# Piper model path
PIPER_MODEL = "/home/mchamp/tts/glados.onnx"



def call_ollama(user_input):
    """Call the Ollama chat API with the given user input and yield responses."""
    global chat_history
    chat_history.append({"role": "user", "content": user_input})
    
    # Prepare the payload with the chat history
    payload = {"model":MODEL_NAME,
                "messages": chat_history,
                "stream":True
                }

    try:
        # The chat endpoint for Ollama
        chat_url = f"{OLLAMA_URL}/api/chat"
        logger.debug(f"sending {payload} to {chat_url}")
        # Make the API request
        response = requests.post(chat_url, json=payload, stream=True)
        response.raise_for_status()  # Ensure the request was successful
        sys.stdout.write("GLaDOS: ")  # This writes a newline before the prompt
        sys.stdout.flush() 
        for line in response.iter_lines():
            if line:  # Skip empty lines
                try:
                    # Parse the JSON response from the stream
                    data = json.loads(line.decode("utf-8"))
                    logger.debug(f"received: {line}")
                    if data.get("done",{}):
                        logger.debug("end of ollama message")
                        sys.stdout.write("\n")  # This writes a newline before the prompt
                        sys.stdout.flush() 
                        yield(".")
                        yield(None)
                    else:
                        content=data.get("message", {}).get("content", "")  # Extract the content  
                        sys.stdout.write(content)
                        sys.stdout.flush()
                    yield content
                except json.JSONDecodeError:
                    continue  # Ignore malformed JSON
        
    except requests.RequestException as e:
        logger.error(f"Error contacting Ollama: {e}")
        yield "Sorry, you seem to have disconnected my brain. Typical."

def call_ollama_generate(prompt):
    """Call Ollama API with the given prompt and yield responses."""
    payload = {"model": MODEL_NAME, "prompt": prompt}
    try:
        chat_url=OLLAMA_URL+"/api/generate"
        response = requests.post(chat_url, json=payload, stream=True)
        response.raise_for_status()  # Ensure the request was successful

        for line in response.iter_lines():
            if line:  # Skip empty lines
                try:
                    # Parse the JSON response from the stream
                    data = json.loads(line.decode("utf-8"))
                    yield data.get("response", "")  # Yield the 'response' field
                except json.JSONDecodeError:
                    continue  # Ignore malformed JSON
    except requests.RequestException as e:
        logger.error(f"Error contacting Ollama: {e}")


def speak_sentence(sentence: str):
    """Speak the given sentence using Piper and aplay."""
    global chat_history
    try:
        stream.start()
        for audio_bytes in voice.synthesize_stream_raw(sentence):
            int_data = np.frombuffer(audio_bytes, dtype=np.int16)
            stream.write(int_data)
        stream.stop()
        chat_history.append({"role": "assistant", "content": sentence})
    except Exception as e:
        logger.error(f"Error during TTS playback: {e}")

def ollama_interface(prompt):
    """Fetch sentences from Ollama and add complete sentences to a queue of things to be spoken."""
    sentence = ""

    for response in call_ollama(prompt):
        if response is None:
            sentence_queue.put(None)
            logger.debug("done receiving ollama")
            break
        else:
            sentence += response
            if sentence and sentence[-1] in ".!?":
                sentence = sentence.replace(":", ".")
                #sentence = re.sub(r"[^A-Za-z0-9\s.!?']", '', sentence) # Remove special characters that cannot be spoken
                sentence_queue.put(sentence)  # Add complete sentence to the queue
                logger.info(f"Queued sentence: {sentence}")
                sentence = ""  # Reset for the next sentence

def tts_thread():
    """Speak sentences from the queue using Piper and aplay."""
    try:
        while True:
            # Wait for an item in the queue
            sentence = sentence_queue.get(block=True, timeout=None)

            if sentence is None:  # Check for sentinel value to exit thread
                logger.debug("finished speaking")
                break

            # Process items in the queue
            speak_sentence(sentence)
            logger.debug(f"speaking: {sentence}")
            sentence_queue.task_done()

    except Exception as e:
        logger.error(f"Error in TTS thread: {e}")
    finally:
        # Clean up and exit the thread
        logger.debug("TTS thread has completed processing and will exit.")


def user_input_thread():
    """Allow user to input text and pass it to Ollama."""
    while True:
        user_input = input("User: ")
        if user_input.strip():  # Only proceed if there's input
            logger.info(f"User input: {user_input}")
            # Pass user input to Ollama thread
            #ollama_interface(user_input)
            threading.Thread(target=ollama_interface, args=(user_input,), daemon=True).start()
            #sys.stdout.write("GLaDOS: ")

if __name__ == "__main__":
    # Start threads
    logging.basicConfig(filename='glados.log', level=logging.WARN)
    logger.info("starting")
    logger.info("loading voice")
    model = "./glados.onnx"
    voice = PiperVoice.load(model)
    logger.debug("model loaded")
    # Setup a sounddevice OutputStream with appropriate parameters
    # The sample rate and channels should match the properties of the PCM data
    stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
    speak_sentence("Initializing")

    logger.info("loading LLM")

    while True:
        user_input = input("Test Subject: ")
        sentence_queue = queue.Queue()
        thread1=threading.Thread(target=ollama_interface, args=(user_input,), daemon=True)
        thread2=threading.Thread(target=tts_thread, daemon=True)
        thread1.start()
        thread2.start()
        sentence_queue.join()
        thread1.join()
        thread2.join()

    # Keep the main thread alive to allow worker threads to run
    try:
        while True:
            pass
    except KeyboardInterrupt:
        stream.close()
        logger.info("Exiting...")
