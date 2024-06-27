import asyncio
import os
import time



names = [
    "en-IE-EmilyNeural",#11223
    "en-US-AvaNeural",#123

]
import edge_tts
def speak(text):
    text=str(text)
    a = time.time()
    command = f'edge-tts --text  "{text}" --voice en-US-AvaNeural --write-media hello.mp3 '
    os.system(command)
    b = time.time()
    print("latency",b-a)


async def speaknew(text):
    time1 = time.time()
    voice = "en-US-AvaNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("hello.mp3")
    time2 = time.time()
    print("text to speech",time2-time1)


