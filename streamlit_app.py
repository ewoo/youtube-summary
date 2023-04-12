from urllib.parse import urlparse
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import whisper
from pytube import YouTube
import openai
from getpass import getpass
from PIL import Image
import requests
from io import BytesIO
import uuid
import os
from urllib.parse import urlparse, parse_qs


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
    div_progress.text('Downloading audio file from Youtube...')
    print("Downloading audio file from Youtube...")
    youtube_video = YouTube(url)

    display_video_stats(youtube_video)

    streams = youtube_video.streams.filter(only_audio=True)
    stream = streams.first()
    stream.download(filename=target_filename)
    return target_filename


def display_video_stats(youtube_video):
    st.write(f'**Title: {youtube_video.title}**')
    duration_in_minutes = round(youtube_video.length / 60)
    st.write(f'Duration: {duration_in_minutes} mins')
    # resp = requests.get(youtube_video.thumbnail_url)
    # video_thumbnail = Image.open(BytesIO(resp.content))
    # st.image(video_thumbnail, caption=youtube_video.title, use_column_width=True)

    # st.write(f'**{youtube_video.description}**')
    # st.write(f'**{youtube_video.views}**')
    # st.write(f'**{youtube_video.rating}**')
    # st.write(f'**{youtube_video.author}**')


def approximate_reading_time(word_count):
    return round(word_count / 200)


def display_transcription_stats(transcript_df):
    # st.write(f'Dialog Content Statistics')
    word_count = transcript_df['token_count'].sum()
    st.write(f"Transcript word count: {word_count}")
    # st.write(
    # f'Aproximate reading time: {approximate_reading_time(word_count)} mins')


def display_summary_stats(token_count):
    # st.write(f'Summary Statistics')
    # st.write(f"Word count: {token_count}")
    mins = approximate_reading_time(token_count)
    if mins <= 1:
        st.write(f'Aproximate reading time: less than a minute')
    else:
        st.write(f'Aproximate reading time: {mins} mins')


@measure_time
def transcribe_audio(model, audio_file):
    div_progress.text('Transcribing audio...')
    print("Transcribing audio...")
    # output = model.transcribe(audio_file)
    audio = open(audio_file, "rb")
    output = openai.Audio.transcribe(
        "whisper-1", audio, resposonse_format="text")
    print("Transcription output:")
    print(output)
    return output


def save_transcription_output(output, txt_file):
    with open(txt_file, 'w', encoding='utf-8') as file:
        file.write(output['text'])
    print(f"Transcription output text saved as {txt_file}.")
    print('')

def load_transcription_file(txt_file):
    transcript = {}
    with open(txt_file, 'r', encoding='utf-8') as file:
        transcript['text'] = file.read()
    return transcript

def generate_narration(summary_text):
    div_progress.text('Generating audio narration...')

    api_key = st.secrets["ELEVENLABS_API_KEY"]
    url = "https://api.elevenlabs.io/v1/text-to-speech/TxGEqnHWrfWFTfGW9XjX"
    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "text": summary_text,
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0
        }
    }

    div_progress.text('Done.')

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            return f"Request failed with status code {response.status_code}"
    except Exception as e:
        return f"Error occurred: {e}"


def get_target_indices(transcript_df, target_sum=4000):
    cumulative_sum = 0
    target_indices = []

    if 'token_count' in transcript_df.columns:
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
    # if(len(target_indices) == 0):
    #     target_indices.append(len(transcript_df) - 1)
    return target_indices


@measure_time
def call_GPT(prompt):
    print("Calling OpenAI...")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=140,
        top_p=1.0,
    )
    return response


def generate_intermmediate_summary(transcript_df, target_indices):
    div_progress.text('Summarizing...')
    responses = []
    summary = ''
    start_index = 0

    if(len(target_indices)>0):
      for end_index in target_indices:
          prompt = transcript_df['text'][start_index: end_index]
          print(f'start:{start_index} end:{end_index}')
          prompt = prompt.astype(str)
          prompt = ''.join(prompt)
          prompt = f"{prompt}\n\ntl;dr:"
          # Call OpenAI.
          resp = call_GPT(prompt)
          summary = summary + resp['choices'][0]['text']
          # Repeat
          start_index = end_index + 1
    else:
        prompt = transcript_df['text']
        print(f'Sending all in one-shot!')
        prompt = prompt.astype(str)
        prompt = ''.join(prompt)
        prompt = f"{prompt}\n\ntl;dr:"
        # Call OpenAI.
        resp = call_GPT(prompt)
        summary = summary + resp['choices'][0]['text']

    return summary


def extract_text_from_response(resp):
    return resp['choices'][0]['text'].strip()


def save_summary(summary, txt_file):
    with open(txt_file, 'w', encoding='utf-8') as file:
        file.write(summary)


def generate_key_takeaways(summary):
    prompt = f"Generate key takeaways:{summary}"
    resp = call_GPT(prompt)
    return resp['choices'][0]['text']

# Main function.
def summarize_youtube_video(youtube_video_url, model):
    # unique_id = str(uuid.uuid4().hex)[:8]

    # extract the query string from the URL
    parsed_url = urlparse(youtube_video_url)
    query_string = parsed_url.query

    # parse the query string into a dictionary
    query_dict = parse_qs(query_string)

    # access the query parameters
    video_name = query_dict['v'][0]

    audio_file = 'audio-' + video_name + '.mp4'
    transcription_file = 'full-transcript-' + video_name + '.txt'
    summary_file = 'summary-' + video_name + '.txt'

    if not os.path.isfile(audio_file):
        download_audio_from_youtube(youtube_video_url, audio_file)
    st.text('Original audio from YouTube video:')
    st.audio(audio_file)

    if not os.path.isfile(transcription_file):
        output = transcribe_audio(model, audio_file)
        save_transcription_output(
            output, transcription_file)
    else:
        output = load_transcription_file(transcription_file)

    # Clean-up audio file.
    # if os.path.exists(audio_file):
    #     os.remove(audio_file)

    print("Converting transcription output to dataframe...")
    print("")

    sentences = output['text'].split('.')
    sentences = [sentence.strip() for sentence in sentences]
    tokens = [len(sentence.split()) for sentence in sentences]

    # transcript_df = pd.DataFrame(output['segments'])
    print(f'sentences:{len(sentences)} tokens:{len(tokens)}')
    transcript_df = pd.DataFrame({'text': sentences, 'token_count': tokens}, index=range(len(sentences)))
    print(transcript_df.head(30))
    # transcript_df['token_count'] = transcript_df['tokens'].apply(len)
    display_transcription_stats(transcript_df)

    # TODO: Chunk summaries only if exceeds target model's token limit.
    # print("Computing target indices...")
    # print("")
    target_indices = get_target_indices(transcript_df)

    div_progress.text('Summarizing transcription...')
    print("Generating summary...")
    print("")
    # summary = generate_intermmediate_summary(transcript_df, [0])
    summary = generate_intermmediate_summary(transcript_df, target_indices)
    save_summary(summary, summary_file)

    print("Generating Key Takeaways...")
    print("")
    final_summary = generate_key_takeaways(summary)

    print("Finished.")
    print("")
    div_progress.text('Done...')

    return summary + "  " + final_summary

# TODO: Add a function to slice the transcript if it's too long.
def slice_if_transcription_is_long(transcript):
    return transcript


@st.cache_resource
def load_openai_whisper_model():
    # One time initialization.
    # Loading model takes a while so should be done once.
    model = whisper.load_model('tiny')
    return model


@st.cache_resource
def initialize_openai_api():
    openai.organization = st.secrets["OPENAI_ORG_ID"]
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    print('Initialized OpenAI API.')
    print('')


def check_if_url_is_valid(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except ValueError:
        return False


# Some global variables. Sorry!
youtube_video = None

# Setup UI.
st.write('## YouTube Podcast Summarizer v0.7')

div_info = st.container()
# Initialize model and OpenAI API.
with div_info:
    st.spinner('Performing Vulcan mind-melding with OpenAI Whisper model...')
    # model = load_openai_whisper_model()
    st.success('Mind-melding complete. OpenAI Whisper model ready.')

initialize_openai_api()
div_header = st.container()
with div_header:
    prompt = "Hi, are you ready for some tasks?"
    st.write(f"**Me:** {prompt}")
    resp = call_GPT(prompt)
    st.write(f'**GPT:** {extract_text_from_response(resp)}')
    st.warning(
        'No GPU enabled. Transcription may be slow. Select shorter videos.', icon='🤖')

model = None

# Get user input.
youtube_video_url = st.text_input(
    'YouTube video URL that you want to summarize:', placeholder='YouTube video URL')
div_button = st.empty()
with div_button:
    submit_button = st.button('Summarize')

if submit_button:
    if check_if_url_is_valid(youtube_video_url):
        div_button.empty()
        with st.spinner('Summarizing your video...'):
            div_progress = st.empty()
            summary = summarize_youtube_video(youtube_video_url, model)
            st.write(f'**Summary**')
            st.write(f'{summary}')
            display_summary_stats(summary)
            
            narration = generate_narration(summary)
            if isinstance(narration, bytes):
                st.text('**Audio Summary**')
                st.audio(narration)
                with open("narration.mp4", "wb") as file:
                    file.write(narration)
                    print("Narration file saved as narration.mp4")
            else:
                print("Error:")
                print(narration)

            # Clear interstatial message area.
            div_progress.empty()
        with div_progress:
            st.success('Completed. Please review the summary below.')
        st.button('Lets Do Another!')
    else:
        with div_info:
            st.warning('Please enter a valid YouTube video URL.')
