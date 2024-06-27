import json
import re
import time

import cv2
import numpy as np
import threading
import tkinter as tk
from PIL import Image, ImageTk
import speechV3
import soundfile as sf
import sounddevice as sd
import pyttsx3
import PIL.Image

from groq import Groq
from faster_whisper import WhisperModel
import pyaudio
import wave
import audioop
import os
import asyncio
def play_here():
    notification_sound_path = "noti.mp3"
    auddata, samra = sf.read(notification_sound_path)
    sd.play(auddata, samra)
    sd.wait()
play_here()
def display_image_on_top(image_path):
    def show_image():
        # Create a new Tkinter window
        window = tk.Tk()
        window.title("Captured Photo")
        window.attributes("-topmost", True)
        window.geometry("+100+100")  # Position the window

        # Load the image
        image = PIL.Image.open(image_path)
        photo = ImageTk.PhotoImage(image)

        # Create a label to hold the image
        label = tk.Label(window, image=photo)
        label.image = photo  # Keep a reference to avoid garbage collection
        label.pack()

        # Run the Tkinter event loop
        window.mainloop()

    # Run the show_image function in a separate thread
    display_thread = threading.Thread(target=show_image)
    display_thread.start()


def click_selfie(p):
    print("GOT UT",p)
    path = "screenshot.png"
    cap = cv2.VideoCapture(0)

    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        return

    # Save the captured frame to a file
    cv2.imwrite(path, frame)

    # Release the webcam
    cap.release()

    # Display the image in a separate window
    display_image_on_top(path)
    img = PIL.Image.open(path)
    # Continue processing with Gemini
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(["answer under 30 words",f"{p}", img])

    return response.text
    #to return


import google.generativeai as genai



import time
import google.generativeai as genai

genai.configure(api_key='************************')
#import PIL.Image
from PIL import ImageGrab
def take_screenshot():
    screenshot = ImageGrab.grab()
    screenshot.save('screenshot.png')
def analyze(p):
    take_screenshot()
    #query  = "solve and answer the question"
    img = PIL.Image.open('screenshot.png')

    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(["answer under 30 words",f"{p}",img])

    return response.text


apiKey = "***********************************************"
client = Groq(api_key=apiKey)
import keyboard
model = WhisperModel('tiny', device="cuda", compute_type="float16")
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD_ENERGY = 2000
THRESHOLD_TIME = 2
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
audio = pyaudio.PyAudio()

#build a project where I can control car racing games using the hand gestures,
#but there was a lot of latency. Can you can you tell me what could have gone wrong?
messages = [
            {
                "role": "system",
                "content": "you are a sarcastic  , humoruos ai assistant that is built be me(ayaan khan) who mocks and roast me every time,give responses under 30 words"
            },

        ]

tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze",
                "description": "get the information about the what is presenton screen",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "p": {
                            "type": "string",
                            "description": "it is the question asked about the data present on screen. e.g.what is the name of actor on screen ",
                        }
                    },
                    "required": ["p"],
                },
            },
        },
{
            "type": "function",
            "function": {
                "name": "click_selfie",
                "description": "clicks the photo using camera and answer the question of visuals",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "p": {
                            "type": "string",
                            "description": "this should be involked when you require vision data from real world E.g what is in my hand , take a photo, what i am holding, take a quick picture of me, what is the colour of my tshirt ",
                        }
                    },
                    "required": ["p"],
                },
            },
        }
    ]


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    # Replace emojis with an empty string
    return emoji_pattern.sub(r'', text)
def speak_async():
    auddata, samra = sf.read("hello.mp3")
    sd.play(auddata, samra)
    sd.wait()
    sd.stop()
    os.remove("hello.mp3")





def listen():

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    start_time = time.time()
    silence_detected = False

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        energy = audioop.rms(data, 2)
        # audio_data = np.frombuffer(data, dtype=np.int16)
        #
        # # Calculate RMS energy
        # energy = np.sqrt(np.mean(audio_data ** 2))

        if energy < THRESHOLD_ENERGY:
            if silence_detected and time.time() - start_time > THRESHOLD_TIME:
                break
            silence_detected = True
        else:
            silence_detected = False
            start_time = time.time()


    time1 = time.time()
    stream.stop_stream()
    stream.close()

    # Save the recorded audio to a file
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # Transcribe the audio
    segments, jvjv = model.transcribe(WAVE_OUTPUT_FILENAME)
    transcription = ""

    for segment in segments:
        transcription += segment.text
    time2 = time.time()
    print("transcribing time: " , (time2-time1))

    return transcription
def get_age(user_type):
    if user_type == "bot":
        return 12
    elif user_type == "my":
        return 19
def process_audio(transcription):
    time1 = time.time()
    if transcription == "exit":
        sd.stop()
    mes = {
                "role": "user",
                "content": f"{transcription}",
            }
    messages.append(mes)

    resource = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.5,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096,

    )

    text = resource.choices[0].message
    ##############################################
    tool_calls = text.tool_calls
    # Step 2: check if the model wanted to call a function
    a = time.time()
    print(tool_calls)
    if tool_calls:
        print("with_tools")
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "analyze": analyze,
            "click_selfie": click_selfie,
        }  # only one function in this example, but you can have multiple
        messages.append(text)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        function_response=""
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                p=function_args.get("p")
            )
            print(function_response)
            print(type(function_response))
            text = function_response

            # extend conversation with functi  on response
          # get a new response from the model where it can see the function response
        text= function_response
        b = time.time()
        print("function calling",b-a)




    ##############################################
    else:
        print("No tool")
        text = text.content
        mes = {
            "role": "assistant",
            "content": f"{text}",
        }
        messages.append(mes)

        print(text)
        time2 = time.time()
        print("Groq Response " , (time2-time1))







    #print("GPT time time: " , (time2-time1))
    #LowLatencyTTS
    #speechV3.speakFast(text)
    #NaturalSoundingTTS
    asyncio.run(speechV3.speaknew(str(text)))
    speak_async()

if __name__ == "__main__":
    #recorder = AudioToTextRecorder(spinner=False, model="tiny.en", language="en")
    while True:
        #print("Listening...")
        if keyboard.is_pressed('shift+x'):
            play_here()

            transcription = listen()
            #transcription = str(input("Enter transcription: "))
            transcription = str(remove_emojis(transcription))
            print(transcription)



            #print("transcription:", transcription,":")
            #print(len(transcription))
            if "exit" in transcription.lower():
                #print("pass1")
                break
            if len(transcription) ==4:
                #print("pass2")
                pass
            else:


                process_audio(str(transcription))





