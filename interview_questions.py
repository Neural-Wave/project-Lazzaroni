import os
from openai import OpenAI
import docx
from InterviewQuestions import load_word_document, question_generator
from TTS_textProcessing import clean_text
from TTS_model import textToWav


key = 'sk-48ixA-rIXOL5Kl-UxBEbIvWikuvBlHWr5Q1zFFD1EyT3BlbkFJHZt-JKu5lLGlWw2L7Hpj8izUK29Dp651YGPzFqPmsA'
#export OPENAI_API_KEY='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'
os.environ["OPENAI_API_KEY"] = key

def load_word_document(file_path):
    """Load a Word document and return its text content."""
    try:
        # Load the document
        doc = docx.Document(file_path)
        
        # Extract text from each paragraph
        text = []
        for para in doc.paragraphs:
            text.append(para.text)

        # Join the list into a single string
        return '\n'.join(text)

    except Exception as e:
        print(f"Error loading document: {e}")
        return None


def welcome_generator(jobdescription):
    client = OpenAI()

    "create an interface to load the documents"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are the hiring responsible. Say hello and welcome the job applicant by its name (Dave) to the job interview process with 1 sentence and ask the applicant how he is doing"},
            {"role": "user", "content": jobdescription }
        ]
    )
    return response.choices[0].message.content

def intro_generator(jobdescription):
    client = OpenAI()

    "create an interface to load the documents"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Introduce that the objective of this interview is to get to better know the person with a 2 sentences."},
            {"role": "user", "content": jobdescription }
        ]
    )
    return response.choices[0].message.content


def question_generator(jobdescription):
    client = OpenAI()

    "create an interface to load the documents"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant and you use the following text to generate 10 questions for an interview of check if the person fullfill the required skills  and Each question should be an individual string in the array."},
            {
                "role": "user",
                "content": jobdescription
            }
        ]
    )
    return response.choices[0].message.content


def wrap_up_generator(jobdescription):
    client = OpenAI()

    "create an interface to load the documents"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Thank the job applicant with one sentence for his time and inform him that he will be contacted within the next days and say goodbye"},
            {"role": "user", "content": jobdescription }
        ]
    )
    return response.choices[0].message.content

# Specify the path to your Word document
file_path = '/teamspace/studios/this_studio/data_JobDescriptions/Fullstack Software Engineer.docx'
    
# Load the document
document_text = load_word_document(file_path)

welcome_text = welcome_generator(document_text)
textToWav(welcome_text, "welcome")

intro_text = intro_generator(document_text)
textToWav(intro_text, "intro_text")

response_text = question_generator(document_text)
questions = [question.strip() for question in response_text.splitlines() if question.strip()]
for i in range(1,len(questions)-1):
    fname = "q" + str(i)
    textToWav(questions[i], fname)

wrap_up = wrap_up_generator(document_text)
textToWav(wrap_up, "wrap_up")

"""
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="Hello Dave, welcome to the job interview process for the Fullstack Engineer position at Acme; how are you doing today?",
                file_path="/teamspace/studios/this_studio/output_voice/Intro_xtts.wav",
                speaker_wav=["/teamspace/studios/this_studio/data_myvoice/DK_1.wav", "/teamspace/studios/this_studio/data_myvoice/DK2.wav",  "/teamspace/studios/this_studio/data_myvoice/DK3.wav",  "/teamspace/studios/this_studio/data_myvoice/DK4.wav", "/teamspace/studios/this_studio/data_myvoice/DK5.wav"],
                language="en",
                enable_text_splitting=False)"""


