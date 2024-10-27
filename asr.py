import time
import torchaudio
from functools import wraps
import logging
import numpy as np
from scipy.signal import resample
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to complete.")
        return result
    return wrapper


def load_openai_model(model_reference_name, gpu_device, automatic_language_detection=False):
    processor = WhisperProcessor.from_pretrained(model_reference_name, force_download=False)
    model = WhisperForConditionalGeneration.from_pretrained(model_reference_name, force_download=False)

    if automatic_language_detection:
        model.config.forced_decoder_ids = None

    if "cuda" in gpu_device.type:
        model = model.to(gpu_device)

    return processor, model


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()
    return sound


class ASR:
    def __init__(self):
        self.clear_transcription_file()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor, self.model = load_openai_model(
            model_reference_name="openai/whisper-base.en",
            gpu_device=self.device,
            automatic_language_detection=False
        )
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
            trust_repo=True
        )
        self.fs = 16_000
        self.chunck = 512
        self.stream = None
        self.audio_buffer = []

        self.speech_active = False
        self.silence_duration = 0.0
        self.speech_duration = 0.0
        self.offset = 0
        self.max_buffer_duration = 30
        self.max_silence_duration = 0.3
        self.finished_talking = False
        self.finished_talking_thres = 5
        self.is_listening = True
        self.whole_transcription = '<<<'

    def clear_transcription_file(self):
        with open('./data_asr/transcription.txt', 'w') as f:
            f.write("")
        logger.info("Transcription file cleared.")

    def save_transcription(self):
        with open('./data_asr/transcription.txt', 'a') as f:
            f.write(self.whole_transcription + "\n\n\n")
        logger.info("Transcription saved to transcription.txt")

    def transcribe(self, new_chunck):
        if self.is_listening:
            transcription = ''
            logger.info(new_chunck)

            sr, y = new_chunck
            if y.ndim > 1:
                y = y.mean(axis=1)
            
            if y.dtype != np.float32:
                y = int2float(y)

            new_samples = int(len(y) * self.fs / sr)
            y = resample(y, new_samples)

            chuncks = [y[i:i + self.chunck] for i in range(0, len(y), self.chunck)]
            n = len(chuncks)
            for m, frame in enumerate(chuncks):
                if len(frame) == self.chunck:
                    frame_t = torch.from_numpy(frame)
                    is_speech =  self.vad_model(frame_t, self.fs).item() > 0.9

                    if is_speech:
                        self.finished_talking = False
                        self.audio_buffer.extend(frame)
                        self.silence_duration = 0 
                        self.speech_active = True
                        self.speech_duration += self.chunck / self.fs
                    else:
                        self.silence_duration += self.chunck / self.fs
                        if self.speech_active and self.silence_duration >= self.max_silence_duration:
                            transcription += self.process_buffered_audio()
                            self.audio_buffer.clear()
                            self.speech_active = False
                            self.speech_duration = 0

                    if self.speech_duration >= self.max_buffer_duration:
                        logger.warn("Max buffer duration exceeded, forcing transcription.")
                        transcription += self.process_buffered_audio()
                        self.audio_buffer.clear()
                        self.speech_active = False
                        self.speech_duration = 0

                    if self.silence_duration >= self.finished_talking_thres:
                        self.finished_talking = True
                        
            logger.info('speech_duration: ' + str(self.speech_duration))
            logger.info('silence_duration: ' + str(self.silence_duration))
            logger.info('transcripion: ' + transcription)
            transcription += '\t\t'
            self.whole_transcription += transcription 
            self.save_transcription()
            return transcription
        else:
            self.finished_talking = False
            return " "
            
    def process_buffered_audio(self):
        t = torch.tensor(self.audio_buffer).unsqueeze(0)
        torchaudio.save('fed.wav', t, self.fs)
        transcription = self.extract_text_from_audio(self.audio_buffer)
        return transcription

    @time_it
    def extract_text_from_audio(self, audio):
        with torch.no_grad():
            prompt = (
                "You need to transcribe voice recordings that are pronounced by a human that is conducting a job interview "
                "for a job developer position. Be careful for term that are informatics related."
            )
            prompt_ids = self.processor.get_prompt_ids(prompt)
            input_features = self.processor(audio, sampling_rate=self.fs, return_tensors="pt").input_features
            predicted_ids = self.model.generate(input_features.to(self.device), temperature=0.2)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return transcription

    def run(self, audio):
        self.transcribe(audio)


def main():
    asr = ASR()


if __name__ == '__main__':
    main()
