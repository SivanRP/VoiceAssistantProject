import sounddevice as sd
import numpy as np
import speech_recognition as sr
import requests
import json
from scipy.io import wavfile
import tempfile
import os

# Cerebras API details
api_key = 'csk-69599dcnpnw3k63k9w6de4yve4p3t8yh5kpwx2pk36dt63p2'
cerebras_url = 'https://api.cerebras.ai/v1/chat/completions'

def process_voice_input():
    try:
        print("Initializing...")
        duration = 5  # seconds
        fs = 44100  # Sample rate
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        print("Recording finished")
        
        # Normalize the recording
        recording = np.int16(recording / np.max(np.abs(recording)) * 32767)
        
        # Save the recording as a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            wavfile.write(temp_wav.name, fs, recording)
        
        # Use Google Speech Recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav.name) as source:
            audio = recognizer.record(source)
        
        # Perform the recognition
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        
        # Clean up the temporary file
        os.unlink(temp_wav.name)
        
    except Exception as e:
        print(f"Error in process_voice_input: {str(e)}")
    
    return None

def process_text_input():
    text = input("Enter your text: ")
    return text

def generate_response(prompt):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'llama3.1-70b',
        'messages': [
            {'role': 'user', 'content': prompt}
        ]
    }
    try:
        print("Sending request to Cerebras API...")
        response = requests.post(cerebras_url, json=data, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        print("Received response from Cerebras API.")
        
        if 'choices' in result and result['choices']:
            return result['choices'][0]['message']['content']
        else:
            print(f"Unexpected response format: {result}")
            return "I'm sorry, I couldn't generate a response."
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return "I'm having trouble connecting to my brain right now."

def speak_response(text):
    print(f"AI says: {text}")

def main():
    while True:
        command = input("Press 'v' for voice input, 't' for text input, or 'q' to quit: ")
        
        if command.lower() == 'q':
            break
        elif command.lower() == 'v':
            user_input = process_voice_input()
        elif command.lower() == 't':
            user_input = process_text_input()
        else:
            print("Invalid command. Please try again.")
            continue
        
        if user_input:
            response = generate_response(user_input)
            speak_response(response)

if __name__ == "__main__":
    main()