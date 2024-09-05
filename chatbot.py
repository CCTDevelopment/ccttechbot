import os
import requests
import sqlite3
import torch
import speech_recognition as sr
import pyttsx3  # For local Text-to-Speech
from gtts import gTTS  # For Google Text-to-Speech (gTTS)
import pyautogui  # For controlling mouse/keyboard actions
import pyperclip  # For clipboard actions
import subprocess  # For running system commands
import time  # For timing delays
import openai  # For GPT-4 interaction
import warnings
import tempfile
import playsound  # To play the generated audio
from transformers import pipeline

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Load OpenAI API key from system environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables.")
openai.api_key = openai_api_key

# Initialize a SQLite database to store memory of past commands and responses
conn = sqlite3.connect('bot_memory.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS memory (query TEXT, response TEXT)''')
conn.commit()

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Initialize a model for text generation with GPU support and set pad_token_id
nlp = None
try:
    nlp = pipeline('text-generation', model='gpt2', device=device, pad_token_id=50256)
    print(f"Model loaded successfully on {'GPU' if device == 0 else 'CPU'}.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Chatbot will continue without model-based responses.")

# Toggle between pyttsx3 and Google TTS
USE_GOOGLE_TTS = True  # Set to True to use Google TTS, False to use pyttsx3

# Text-to-Speech (local TTS using pyttsx3)
def speak_text(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    # Set voice to female (optional)
    for voice in voices:
        if 'female' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    engine.setProperty('rate', 150)  # Set the speech rate
    engine.say(text)
    engine.runAndWait()

# Alternatively, use Google TTS for more natural voice output (requires internet)
def gtts_speak(text):
    tts = gTTS(text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(fp.name + ".mp3")
        playsound.playsound(fp.name + ".mp3")

# Run system commands
def run_system_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Failed to run the command: {str(e)}"

# Store command and response in memory
def store_in_memory(query, response):
    c.execute("INSERT INTO memory (query, response) VALUES (?, ?)", (query, response))
    conn.commit()

# Check if a similar query is already in memory
def check_memory(query):
    c.execute("SELECT response FROM memory WHERE query LIKE ?", ('%' + query + '%',))
    result = c.fetchone()
    return result[0] if result else None

# Function to type or paste text into an open window (e.g., Visual Studio Code)
def type_into_window(text):
    pyperclip.copy(text)  # Copy the text to the clipboard
    pyautogui.hotkey('ctrl', 'v')  # Paste the clipboard content
    pyautogui.press('enter')  # Press enter to execute the command

# Function to open VSCode and insert code
def open_vscode_and_insert_code(code):
    # Open VSCode (for Linux; adjust for Windows/Mac)
    run_system_command("code &")
    time.sleep(3)  # Wait for VSCode to open

    # Type the generated code into VSCode
    type_into_window(code)

# Function to ask questions and generate code with GPT-4
def ask_questions_for_code():
    # You can modify this to ask relevant questions
    questions = [
        "What kind of application are you building?",
        "Which programming language would you like to use?",
        "Do you need any specific features (e.g., database, user authentication)?"
    ]
    
    answers = []
    for question in questions:
        print(f"Question: {question}")
        answer = input("Your answer: ")
        answers.append(answer)
    
    # Ask GPT-4 to generate code based on the answers
    prompt = f"You are a highly skilled developer. Create a {answers[0]} in {answers[1]} that includes {answers[2]}."
    code_response = interact_with_gpt4(prompt)
    
    # Insert the generated code into VSCode
    open_vscode_and_insert_code(code_response)

# Updated GPT-4 interaction function
def interact_with_gpt4(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a highly intelligent system that assists with Linux commands and development. Output only code or commands without explanations, commentary, or any additional text. "},
                {"role": "user", "content": query}
            ],
            max_tokens=300
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Failed to get response from GPT-4: {str(e)}"

# Function to capture dictation from the microphone (continuous listening)
def capture_dictation():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 200  # Adjust threshold
    full_text = ""
    try:
        with sr.Microphone(device_index=0) as source:
            print("Dictation mode: Speak now. Say 'stop dictation' to end.")
            while True:
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    text = recognizer.recognize_google(audio)
                    if 'stop dictation' in text.lower():
                        print("Dictation mode stopped.")
                        break
                    print(f"Dictated: {text}")
                    full_text += text + "\n"
                except sr.UnknownValueError:
                    print("Sorry, I didn't catch that.")
                except sr.RequestError as e:
                    print(f"Could not request results from Google STT; {e}")
    except AssertionError as e:
        print(f"Microphone error: {str(e)}")
    return full_text

# Function to capture voice input from the microphone (single command mode)
def capture_audio():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 200  # Adjust threshold
    try:
        with sr.Microphone(device_index=0) as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Audio captured, processing...")
            try:
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that.")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from Google STT; {e}")
                return None
    except AssertionError as e:
        print(f"Microphone error: {str(e)}")
        return None

# Main chatbot function
def chatbot(query):
    # Check internal memory first
    memory_response = check_memory(query)
    if memory_response:
        return memory_response

    # Handle dictation mode
    if 'start dictation' in query.lower():
        capture_dictation()  # Start dictation mode and capture speech
        return "Dictation mode started."

    # Handle code writing in VSCode
    if 'write code' in query.lower():
        ask_questions_for_code()  # Ask the user questions and generate code in VSCode
        return "Code writing mode started."

    # Handle command execution
    if 'install' in query.lower():
        command = f"sudo apt-get install -y {query.split('install')[-1].strip()}"
        response = run_system_command(command)
        store_in_memory(query, response)  # Store the response in memory
        return response

    if 'run' in query.lower():  # Generic run command handler
        command = query.lower().replace('run', '').strip()
        response = run_system_command(command)
        store_in_memory(query, response)  # Store the response in memory
        return f"Command output:\n{response}"

    # Ask GPT-4 if no internal memory or command found
    gpt_response = interact_with_gpt4(query)
    store_in_memory(query, gpt_response)  # Store the GPT-4 response in memory
    return gpt_response

# Example of chatbot interaction loop with voice input and output
if __name__ == "__main__":
    print("CCT Tech Bot is ready! Say 'exit' to stop.")
    while True:
        user_input = capture_audio()
        if user_input and user_input.lower() == 'exit':
            print("CCT Tech Bot terminated.")
            break
        if user_input:
            bot_response = chatbot(user_input)
            print(f"Bot: {bot_response}")
            
            # Choose between pyttsx3 and gtts for voice output based on toggle
            if USE_GOOGLE_TTS:
                gtts_speak(bot_response)  # Use Google TTS for natural voice
            else:
                speak_text(bot_response)  # Use pyttsx3 for local TTS
