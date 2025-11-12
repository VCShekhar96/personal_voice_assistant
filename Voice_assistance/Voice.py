import os
import time
import datetime
import webbrowser
import smtplib
import speech_recognition as sr
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import librosa
from scipy.spatial.distance import cosine
import warnings
import threading
import subprocess
from email.mime.text import MIMEText
import requests
from bs4 import BeautifulSoup
import nltk
import pygame
from datetime import timedelta
import json
from googletrans import Translator, LANGUAGES

# Download NLTK resources at startup
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Suppress FutureWarning for librosa
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 0.9)

# Initialize pygame for music playback
pygame.mixer.init()

# Initialize translator
translator = Translator()

# File paths for persistence
PREFS_FILE = "user_prefs.json"
DATA_FILE = "user_data.json"
QUICK_COMMANDS_FILE = "quick_commands.json"

# Load or initialize user preferences
def load_preferences():
    try:
        if os.path.exists(PREFS_FILE):
            with open(PREFS_FILE, "r") as f:
                return json.load(f)
        return {"language": "en", "temp_unit": "C", "music_genre": "pop", "retain_data": True}
    except Exception as e:
        print(f"Error loading preferences: {e}")
        return {"language": "en", "temp_unit": "C", "music_genre": "pop", "retain_data": True}

# Save user preferences
def save_preferences(prefs):
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f)
    except Exception as e:
        print(f"Error saving preferences: {e}")

# Load or initialize user data
def load_user_data():
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        return {"queries": []}
    except Exception as e:
        print(f"Error loading user data: {e}")
        return {"queries": []}

# Save user data
def save_user_data(data):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving user data: {e}")

# Load or initialize quick commands
def load_quick_commands():
    try:
        if os.path.exists(QUICK_COMMANDS_FILE):
            with open(QUICK_COMMANDS_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading quick commands: {e}")
        return {}

# Save quick commands
def save_quick_commands(commands):
    try:
        with open(QUICK_COMMANDS_FILE, "w") as f:
            json.dump(commands, f)
    except Exception as e:
        print(f"Error saving quick commands: {e}")

# Expanded knowledge base for concise answers (under 100 words)
knowledge_base = {
    "what is python": "Python is a versatile, high-level programming language used for web development, data science, and automation.",
    "define artificial intelligence": "AI is the simulation of human intelligence by machines, enabling learning and problem-solving.",
    "what is gravity": "Gravity is the force that attracts objects towards each other, causing them to come together or move closer.",
    "capital of india": "The capital of India is New Delhi.",
    "capital of usa": "The capital of the United States is Washington, D.C.",
    "capital of france": "The capital of France is Paris.",
    "who is the prime minister of india": "Narendra Modi is the Prime Minister of India, serving since May 26, 2014.",
    "who is the president of america": "Donald J. Trump is the President of the United States, serving since January 20, 2025.",
    "who is the president of usa": "Donald J. Trump is the President of the United States, serving since January 20, 2025.",
    "who invented electricity": "Michael Faraday's work in the 1830s on electromagnetic induction enabled practical use of electricity.",
    "what is 2 plus 2": "Two plus two equals four."
}

# Context storage for follow-up questions and intents
conversation_context = {"last_topic": None, "last_intent": None, "history": []}

# Local calendar storage
calendar_events = []

# User preferences, data, and quick commands
user_prefs = load_preferences()
user_data = load_user_data()
quick_commands = load_quick_commands()

# Function to speak text with language support
def speak(text, lang=user_prefs["language"]):
    try:
        if lang != "en":
            text = translator.translate(text, dest=lang).text
        print(f"Assistant: {text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# Function to record audio
def record_audio(file_path, duration=5, sr=16000):
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()
        wavfile.write(file_path, sr, audio)
        return True
    except Exception as e:
        print(f"Error recording audio: {e}")
        return False

# Function to load audio
def load_voice_sample(file_path, sr=16000):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file {file_path} not found.")
        audio, _ = librosa.load(file_path, sr=sr)
        return audio
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

# Function to compare voices using MFCC
def compare_voices_mfcc(reference_file, live_file, sr=16000):
    try:
        ref_audio = load_voice_sample(reference_file, sr)
        live_audio = load_voice_sample(live_file, sr)
        if ref_audio is None or live_audio is None:
            return False
        mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=13)
        mfcc_live = librosa.feature.mfcc(y=live_audio, sr=sr, n_mfcc=13)
        similarity = 1 - cosine(mfcc_ref.flatten(), mfcc_live.flatten())
        threshold = 0.7
        print(f"MFCC similarity: {similarity:.2f}")
        return similarity > threshold
    except Exception as e:
        print(f"Error in MFCC comparison: {e}")
        return False

# Function to detect basic emotional tone
def detect_tone(file_path, sr=16000):
    try:
        audio = load_voice_sample(file_path, sr)
        if audio is None:
            return "neutral"
        pitch = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
        avg_pitch = np.mean(pitch)
        if avg_pitch > 200:
            return "excited"
        elif avg_pitch < 100:
            return "calm"
        return "neutral"
    except Exception as e:
        print(f"Error in tone detection: {e}")
        return "neutral"

# Function to authenticate voice
def authenticate_voice(reference_file="user_voice_sample.wav", live_file="live_voice.wav"):
    if not os.path.exists(reference_file):
        speak("Record a voice sample by saying 'Hey'.")
        if not record_audio(reference_file):
            speak("Failed to record voice sample.")
            return False
    speak("Say 'Hey' to authenticate.")
    if not record_audio(live_file):
        speak("Failed to record live voice.")
        return False
    if compare_voices_mfcc(reference_file, live_file):
        tone = detect_tone(live_file)
        speak(f"Authentication successful! You sound {tone}.")
        return True
    else:
        speak("Authentication failed.")
        return False

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            command = recognizer.recognize_google(audio, language=user_prefs["language"]).lower()
            print(f"Command: {command}")
            if user_prefs["retain_data"]:
                user_data["queries"].append({"command": command, "timestamp": datetime.datetime.now().isoformat()})
                save_user_data(user_data)
            return command
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            speak("Sorry, I didn't understand that.")
            return ""
        except sr.RequestError as e:
            speak(f"Speech recognition error: {e}")
            return ""

# Function to answer questions concisely
def answer_question(question):
    global conversation_context
    question = question.strip().lower()

    # Update conversation history
    conversation_context["history"].append(question)
    if len(conversation_context["history"]) > 5:
        conversation_context["history"].pop(0)

    # Check local knowledge base
    for key in knowledge_base:
        if key in question:
            conversation_context["last_topic"] = key
            conversation_context["last_intent"] = "question"
            return knowledge_base[key]

    # Web-based answer
    try:
        query = question.replace(" ", "+")
        url = f"https://en.wikipedia.org/w/index.php?search={query}"
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 20:
                conversation_context["last_topic"] = question
                conversation_context["last_intent"] = "question"
                return text[:80] + "..."
    except Exception as e:
        print(f"Web search error: {e}")

    # Fallback to Google search
    webbrowser.open(f"https://www.google.com/search?q={query}")
    conversation_context["last_topic"] = question
    conversation_context["last_intent"] = "question"
    return f"Opened a search for {question}."

# Function to get weather (offline fallback)
def get_weather(city):
    try:
        api_key = "your_openweathermap_api_key"  # Replace with your API key
        unit = "metric" if user_prefs["temp_unit"] == "C" else "imperial"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units={unit}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data["cod"] == 200:
            temp = data["main"]["temp"]
            description = data["weather"][0]["description"]
            unit_symbol = "°C" if unit == "metric" else "°F"
            return f"{city}: {temp}{unit_symbol}, {description}."
        else:
            return f"Couldn't fetch weather for {city}."
    except Exception as e:
        print(f"Weather API error: {e}")
        return "Error fetching weather. Try again offline."

# Function to control smart devices (placeholder)
def control_smart_device(device, action):
    speak(f"Placeholder: {action} {device}. Configure a smart home API.")

# Function to set a reminder
def set_reminder(description, minutes, email=False, to_email=None):
    event_time = datetime.datetime.now() + timedelta(minutes=minutes)
    calendar_events.append({"description": description, "time": event_time})
    speak(f"Reminder set for '{description}' at {event_time.strftime('%I:%M %p')}.")
    if email and to_email:
        send_email(to_email, "Reminder", f"Reminder: {description} at {event_time.strftime('%I:%M %p')}")
    threading.Thread(target=check_reminder, args=(description, event_time), daemon=True).start()

# Function to check reminders
def check_reminder(description, event_time):
    while datetime.datetime.now() < event_time:
        time.sleep(10)
    speak(f"Reminder: {description}")

# Function to play music (offline)
def play_music(file_path=None):
    try:
        if not file_path:
            file_path = f"D:/Music/{user_prefs['music_genre']}_sample.mp3"  # Update path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Music file {file_path} not found.")
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        speak(f"Playing {user_prefs['music_genre']} music.")
    except Exception as e:
        speak(f"Error playing music: {e}")

# Function to stop music
def stop_music():
    pygame.mixer.music.stop()
    speak("Music stopped.")

# Function to set a timer (offline)
def set_timer(minutes):
    speak(f"Setting a timer for {minutes} minutes.")
    seconds = minutes * 60
    time.sleep(seconds)
    speak("Timer finished!")

# Function to send an email
def send_email(to_email, subject, body):
    try:
        from_email = "your_email@gmail.com"  # Replace with your email
        password = "your_app_password"  # Replace with your app password
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        speak("Email sent successfully!")
    except Exception as e:
        speak(f"Failed to send email: {e}")

# Function for Live Speech (accessibility)
def live_speech():
    speak("Say or type what you want to speak.")
    text = recognize_speech()
    if text:
        speak(f"Speaking: {text}")
        return text
    else:
        speak("No input detected.")
        return ""

# Function for Text Call simulation
def text_call():
    speak("Simulating incoming call. Speak to transcribe or say 'type' to respond.")
    response = recognize_speech()
    if response == "type":
        speak("Say your response to type.")
        typed_response = recognize_speech()
        if typed_response:
            speak(f"Responding: {typed_response}")
        else:
            speak("No response provided.")
    elif response:
        speak(f"Transcribed: {response}. Say 'type' to respond or continue speaking.")
    else:
        speak("No input detected.")

# Function to translate text (Galaxy AI-like)
def translate_text(text, target_lang):
    try:
        translated = translator.translate(text, dest=target_lang).text
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return "Error translating text."

# Function to set quick command
def set_quick_command(name, actions):
    quick_commands[name] = actions
    save_quick_commands(quick_commands)
    speak(f"Quick command '{name}' set with {len(actions)} actions.")

# Function to execute quick command
def execute_quick_command(name, authenticated):
    if name not in quick_commands:
        speak(f"Quick command '{name}' not found.")
        return
    for action in quick_commands[name]:
        process_command(action, authenticated)

# Function to check routines (time-based)
def check_routines():
    while True:
        current_time = datetime.datetime.now().strftime("%H:%M")
        for event in calendar_events:
            if event["time"].strftime("%H:%M") == current_time:
                speak(f"Routine: {event['description']}")
        time.sleep(60)

# Function to set routine
def set_routine(description, hour, minute):
    event_time = datetime.datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    if event_time < datetime.datetime.now():
        event_time += timedelta(days=1)
    calendar_events.append({"description": description, "time": event_time})
    speak(f"Routine set for '{description}' at {event_time.strftime('%I:%M %p')} daily.")

# Function to delete user data
def delete_user_data():
    global user_data
    user_data = {"queries": []}
    save_user_data(user_data)
    speak("User data deleted.")

# Function to process commands
def process_command(command, authenticated=False):
    global conversation_context, user_prefs
    try:
        tokens = nltk.word_tokenize(command)
        tagged = nltk.pos_tag(tokens)
    except LookupError as e:
        speak("Error with NLP resources. Please ensure NLTK data is downloaded.")
        print(f"NLTK error: {e}")
        tokens = command.split()
        tagged = []

    # Update conversation history
    conversation_context["history"].append(command)
    if len(conversation_context["history"]) > 5:
        conversation_context["history"].pop(0)

    # Check quick commands
    for qc_name in quick_commands:
        if qc_name.lower() in command:
            execute_quick_command(qc_name.lower(), authenticated)
            return True

    if "hi buddy" in command:
        speak("Hello! I'm Buddy, your voice assistant. How can I help you?")
        return True
    elif "time" in command:
        current_time = datetime.datetime.now().strftime("%I:%M %p, %B %d, %Y")
        speak(f"The time is {current_time}.")
        conversation_context["last_intent"] = "time"
    elif "date" in command:
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        speak(f"Today is {current_date}.")
        conversation_context["last_intent"] = "date"
    elif "open" in command:
        if "google" in command:
            webbrowser.open("https://www.google.com")
            speak("Opening Google.")
        elif "youtube" in command:
            webbrowser.open("https://www.youtube.com")
            speak("Opening YouTube.")
        elif "notepad" in command:
            subprocess.Popen("notepad.exe")
            speak("Opening Notepad.")
        else:
            speak("Sorry, I don't know how to open that.")
        conversation_context["last_intent"] = "open"
    elif "search" in command:
        query = command.replace("search", "").strip()
        webbrowser.open(f"https://www.google.com/search?q={query}")
        speak(f"Searching for {query}.")
        conversation_context["last_intent"] = "search"
    elif "timer" in command:
        try:
            minutes = int(command.split("timer")[1].split("minute")[0].strip())
            threading.Thread(target=set_timer, args=(minutes,), daemon=True).start()
        except (IndexError, ValueError):
            speak("Please specify the timer duration in minutes.")
        conversation_context["last_intent"] = "timer"
    elif "email" in command:
        if not authenticated:
            speak("Please authenticate to send an email.")
            return True
        speak("Provide the recipient's email address.")
        to_email = recognize_speech()
        if to_email:
            speak("What is the subject?")
            subject = recognize_speech()
            if subject:
                speak("What is the body?")
                body = recognize_speech()
                if body:
                    send_email(to_email, subject, body)
                else:
                    speak("No email body provided.")
            else:
                speak("No subject provided.")
        else:
            speak("No recipient email provided.")
        conversation_context["last_intent"] = "email"
    elif "weather" in command:
        city = command.replace("weather", "").strip()
        if city:
            speak(get_weather(city))
        else:
            speak("Please specify a city.")
        conversation_context["last_intent"] = "weather"
    elif "reminder" in command:
        speak("What is the reminder about?")
        description = recognize_speech()
        if description:
            speak("In how many minutes?")
            minutes_str = recognize_speech()
            try:
                minutes = int(minutes_str)
                email = "email reminder" in command
                to_email = None
                if email:
                    speak("Provide the recipient's email.")
                    to_email = recognize_speech()
                set_reminder(description, minutes, email, to_email)
            except ValueError:
                speak("Specify a valid number of minutes.")
        else:
            speak("No reminder description provided.")
        conversation_context["last_intent"] = "reminder"
    elif "play music" in command:
        play_music()
        conversation_context["last_intent"] = "music"
    elif "stop music" in command:
        stop_music()
        conversation_context["last_intent"] = "music"
    elif "smart home" in command or "control" in command:
        device = "light" if "light" in command else "thermostat"
        action = "turn on" if "on" in command else "turn off"
        control_smart_device(device, action)
        conversation_context["last_intent"] = "smart_home"
    elif "live speech" in command:
        live_speech()
        conversation_context["last_intent"] = "live_speech"
    elif "text call" in command:
        text_call()
        conversation_context["last_intent"] = "text_call"
    elif "translate" in command:
        speak("What text to translate?")
        text = recognize_speech()
        if text:
            speak("To which language?")
            lang = recognize_speech()
            lang_code = None
            for code, name in LANGUAGES.items():
                if name.lower() in lang.lower():
                    lang_code = code
                    break
            if lang_code:
                translated = translate_text(text, lang_code)
                speak(f"Translated: {translated}")
            else:
                speak("Language not recognized.")
        else:
            speak("No text provided.")
        conversation_context["last_intent"] = "translate"
    elif "set quick command" in command:
        speak("What is the quick command name?")
        name = recognize_speech()
        if name:
            speak("Say the actions, one by one. Say 'done' when finished.")
            actions = []
            while True:
                action = recognize_speech()
                if action == "done":
                    break
                if action:
                    actions.append(action)
            if actions:
                set_quick_command(name, actions)
            else:
                speak("No actions provided.")
        else:
            speak("No command name provided.")
        conversation_context["last_intent"] = "quick_command"
    elif "set routine" in command:
        speak("What is the routine about?")
        description = recognize_speech()
        if description:
            speak("At what hour? Say a number from 0 to 23.")
            hour_str = recognize_speech()
            speak("At what minute?")
            minute_str = recognize_speech()
            try:
                hour = int(hour_str)
                minute = int(minute_str)
                set_routine(description, hour, minute)
            except ValueError:
                speak("Specify valid hour and minute.")
        else:
            speak("No routine description provided.")
        conversation_context["last_intent"] = "routine"
    elif "set language" in command:
        lang_query = command.replace("set language", "").strip()
        lang_code = None
        for code, name in LANGUAGES.items():
            if name.lower() in lang_query.lower():
                lang_code = code
                break
        if lang_code:
            user_prefs["language"] = lang_code
            save_preferences(user_prefs)
            speak(f"Language set to {LANGUAGES[lang_code]}.", lang="en")
        else:
            speak("Language not recognized. Try 'set language to French'.")
        conversation_context["last_intent"] = "language"
    elif "set temperature unit" in command:
        unit = "C" if "celsius" in command else "F" if "fahrenheit" in command else None
        if unit:
            user_prefs["temp_unit"] = unit
            save_preferences(user_prefs)
            speak(f"Temperature unit set to {unit}.")
        else:
            speak("Specify Celsius or Fahrenheit.")
        conversation_context["last_intent"] = "prefs"
    elif "set music genre" in command:
        genre = command.replace("set music genre", "").strip()
        if genre:
            user_prefs["music_genre"] = genre
            save_preferences(user_prefs)
            speak(f"Music genre set to {genre}.")
        else:
            speak("Specify a music genre.")
        conversation_context["last_intent"] = "prefs"
    elif "set data retention" in command:
        retain = "enable" in command or "on" in command
        user_prefs["retain_data"] = retain
        save_preferences(user_prefs)
        speak(f"Data retention {'enabled' if retain else 'disabled'}.")
        if not retain:
            delete_user_data()
        conversation_context["last_intent"] = "prefs"
    elif "delete data" in command:
        delete_user_data()
        conversation_context["last_intent"] = "prefs"
    elif "shutdown" in command:
        speak("Shutting down system in 30 seconds.")
        os.system("shutdown /s /t 30")
        conversation_context["last_intent"] = "system"
    elif "cancel shutdown" in command:
        os.system("shutdown /a")
        speak("Shutdown canceled.")
        conversation_context["last_intent"] = "system"
    elif "exit" in command or "stop" in command:
        speak("Goodbye!")
        return False
    elif command:
        if any(word in command for word in ["what", "who", "where", "when", "why", "how", "define", "capital"]):
            answer = answer_question(command)
            speak(answer)
            conversation_context["last_intent"] = "question"
        elif "more" in command and conversation_context["last_topic"]:
            speak(f"More information about {conversation_context['last_topic']}.")
            answer = answer_question(conversation_context["last_topic"])
            speak(answer)
            conversation_context["last_intent"] = "question"
        else:
            if conversation_context["last_intent"] == "question" and any(word in command for word in ["he", "she", "they", "it"]):
                answer = answer_question(conversation_context["last_topic"] + " " + command)
                speak(answer)
            else:
                speak("I didn't understand. Try a question or another command.")
        conversation_context["last_intent"] = "unknown"
    return True

# Main function
def main():
    # Start routine checker
    threading.Thread(target=check_routines, daemon=True).start()

    speak("Voice assistant starting. Initializing authentication.")
    authenticated = authenticate_voice()
    if not authenticated:
        speak("Authentication failed. Exiting.")
        return

    speak("Authentication successful. Say 'Hi Buddy' to activate me.")
    running = True
    while running:
        command = recognize_speech()
        if command:
            running = process_command(command, authenticated)

if __name__ == "__main__":
    main()