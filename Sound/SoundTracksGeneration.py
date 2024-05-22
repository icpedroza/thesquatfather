import os

from gtts import gTTS

from playsound import playsound

def textVoice(givenText):
    #test = "Hello"
    tts = gTTS(text=givenText, lang="en")
    tts.save("Sound.mp3")
    playsound("Sound.mp3")
    os.remove('Sound.mp3')
'''text = str(input())
textVoice(text)'''