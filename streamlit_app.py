import streamlit as st
import pandas as pd
import numpy as np

import datetime
import pickle
import whisper
from pytube import YouTube
import openai
from getpass import getpass

def measure_time(func):
    def wrapper(*args, **kwargs):
        t1 = datetime.datetime.now()
        print(f"Started at {t1}")

        result = func(*args, **kwargs)

        print(f"Done.")
        t2 = datetime.datetime.now()
        print(f"Ended at {t2}")
        print(f"Elapsed time: {t2 - t1}")
        print()
        return result
    return wrapper

@measure_time
def download_audio_from_youtube(url, target_filename):
    print("Downloading audio file from Youtube...")
    youtube_video = YouTube(url)
    streams = youtube_video.streams.filter(only_audio=True)
    stream = streams.first()
    stream.download(filename=target_filename)
    return target_filename

@measure_time
def transcribe_audio(model, audio_file):
    print("Transcribing audio...")
    output = model.transcribe(audio_file)
    return output

def save_transcription_output(output, pkl_file, txt_file):
    with open(pkl_file, 'wb') as file:
        pickle.dump(output, file)
    print(f"Transcription output saved as {pkl_file}.")

    with open(txt_file, 'w', encoding='utf-8') as file:
        file.write(output['text'])
    print(f"Transcription output text saved as {txt_file}.")
    print('')

def get_target_indices(transcript_df, target_sum=4097):
    cumulative_sum = 0
    target_indices = []

    for index, row in transcript_df.iterrows():
        cumulative_sum += row['token_count']
        if cumulative_sum >= target_sum:
            target_indices.append(index)
            cumulative_sum = 0
    # This function causes bugs. It seems token count is not consistent
    # with what the OpenAI expects. For example, when using Korean
    # the API rejected my prompt saying it was over the 4097 token limit
    # (when it wasn't.)

    # Also, put a default and slice only if entire chunk is larger than
    # token limit of 4097.
    # if(len(target_indices)==0):
    #     target_indices.append(len(transcript_df) - 1)
    return target_indices

@measure_time
def generate_response(prompt):  
    print("Calling OpenAI...")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=140,
        top_p=1.0,
    )
    return response

def generate_summary(transcript_df, target_indices):
    responses = []
    summary = ''
    start_index = 0

    for end_index in target_indices:
        prompt = transcript_df['text'][start_index: end_index]
        print(f'start:{start_index} end:{end_index}')
        prompt = prompt.astype(str)
        prompt = ''.join(prompt)
        prompt = f"{prompt}\n\ntl;dr:"

        # Call OpenAI.
        resp = generate_response(prompt)
        summary = summary + resp['choices'][0]['text']

        start_index = end_index + 1

    return summary

def save_summary(summary, txt_file):
    with open(txt_file, 'w', encoding='utf-8') as file:
        file.write(summary)

def generate_summary_of_summaries(summary):
    prompt = f"{summary}\n\ntl;dr:"
    resp = generate_response(prompt)
    return resp['choices'][0]['text']

def summarize_youtube_video(youtube_video_url):
  youtube_video_url = youtube_video_url
  audio_file = 'audio.mp4'
  download_audio_from_youtube(youtube_video_url, audio_file)

  output = transcribe_audio(model, audio_file)
  save_transcription_output(output, 'audio_transcription.pkl', 'audio_transcription.txt')

  transcript_df = pd.DataFrame(output['segments'])
  transcript_df['token_count'] = transcript_df['tokens'].apply(len)
  transcript_df.head()

  target_indices = get_target_indices(transcript_df)
  summary = generate_summary(transcript_df, target_indices)
  save_summary(summary, 'episode_summary.txt')

  final_summary = summary
  if len(target_indices) > 3:
    final_summary = generate_summary_of_summaries(summary)

  return final_summary

# Setup UI.
st.title('YouTube Podcast Summarizer')

# One time initialization.
model = whisper.load_model('base') # Loading model takes a while so should be done once.

# Configuration specific to me.
# Setup OpenAI API. 
openai.organization = "org-vuYlHZXjJF5eEOed6foak12t" # This orgnization ID is for Wooyong Ee!
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompt for secret key.
openai.api_key = getpass() # Enter my OpenAPI API secret key
