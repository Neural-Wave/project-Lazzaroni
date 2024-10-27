from TTS.api import TTS
from TTS_textProcessing import clean_text
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)


def textToWav(text, name):# generate speech by cloning a voice using default settings
    text= clean_text(text)
    print(text)
    tts.tts_to_file(text= text,
                file_path="/teamspace/studios/this_studio/output_voice/"+ name + ".wav",
                speaker_wav=["/teamspace/studios/this_studio/data_myvoice/DK_1.wav", "/teamspace/studios/this_studio/data_myvoice/DK2.wav",  "/teamspace/studios/this_studio/data_myvoice/DK3.wav",  "/teamspace/studios/this_studio/data_myvoice/DK4.wav", "/teamspace/studios/this_studio/data_myvoice/DK5.wav"],
                language="en",
                enable_text_splitting=False)

