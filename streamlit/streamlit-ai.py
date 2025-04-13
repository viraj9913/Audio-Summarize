import os
from openai import OpenAI
import requests
import streamlit as st
import whisper
import openai
from pydub import AudioSegment


HUGGINGFACE_API_KEY = st.secrets["HF_API_TOKEN"]

# Define the Hugging Face inference endpoint
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

# Set up headers with API key
headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
}

# //////

# chunking each file text
def chunktext(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # the last sentence should end with . or ? or !
        if end < len(text):
            while end > start and text[end] not in '.!?':
                end -= 1
        chunks.append(text[start:end])
        # what should be the next starting point after chunk 1 is passed
        # to continue, it should find the word where it cuts off and overlap by 500 so that it will lookup for previous 500 words cut from the specific word
        start += chunk_size - overlap
    return chunks

def getsummary(content):
    payload = {"inputs": content}
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    result = response.json()
    print("API result:", result)
    if isinstance(result, list):
        return result[0]["summary_text"]
    else:
        return "Error in summarization: " + str(result)

# passing the text
def summarizetext(filepath):
    # # reading the file
    # with open(filepath, 'r', encoding='utf-8') as file:
    #     content = file.read()

    # limited words will be passed, for prevention of hallucination
    chunks = chunktext(filepath)
    
    summaries = []
    # collecting the summary
    for chunk in chunks:
        summary = getsummary(chunk)
        summaries.append(summary)
    
    # returning each chunks to file
    finalsummary = "\n".join(summaries)
    return finalsummary


def audiotxt(file):
    model = whisper.load_model("small")

    # Convert uploaded file to a format Whisper understands
    audio = AudioSegment.from_file(file)
    audio.export("temp.wav", format="wav")  # Save to a temporary file

    result = model.transcribe("temp.wav")
    return result["text"]

# /\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#  void main()

st.title("AI Podcast Summarizer (Mistral via Ollama)")

user_input = st.text_area("Paste transcript or summary chunk")

st.header('Or Upload a wav audio file')

file=st.file_uploader('Choose a file (m4a).',type=['m4a'])

# if user selects file to upload
if file is not None:
    st.write('Processing {}m4a file'.format(file))
    with st.spinner('Wait for it...',show_time=True):

        text = audiotxt(file)
        output = summarizetext(text)
        st.markdown("### Summary")
        st.write(output)
        # button to download the summary file
        st.download_button('Download summary',output,file_name='summary.txt')
        
        # clean up the temp.wav file, stored in the memory
        os.remove('temp.wav') 


# # if user wants to directly copy paste the content
# if st.button("Summarize"):
#     with st.spinner('Wait for it...',show_time=True):
#         output = summarizetext(user_input)
#         st.markdown("### Summary")
#         st.download_button('Download summary',output,file_name='summary.txt')


if st.button("Summarize"):
    if not user_input.strip():
        st.warning("Please enter some text to summarize.")
    else:
        # st.write("Input received! Length:", len(user_input))
        chunks = chunktext(user_input)
        # st.write(f"Generated {len(chunks)} chunk(s).")
        # st.write("First chunk preview:", chunks[0] if chunks else "None")

        with st.spinner('Summarizing...'):
            if chunks:
                output = summarizetext(user_input)
                st.markdown("### Summary")
                st.write(output)
                st.download_button('Download summary', output, file_name='summary.txt')
            else:
                st.error("Text is too short or malformed for summarization. Try pasting a longer transcript.")

    




# for mistral model
# took 1 min and 40 seconds to complete the audio
# for text took 1 min and 30 seconds


# for wizardlm2
# for audio 2 min and 34 seconds
# for text took 1 min 
