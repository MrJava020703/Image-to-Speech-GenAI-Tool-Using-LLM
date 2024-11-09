import os
import time
import openai
import requests
import streamlit as st
from typing import Any
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import pipeline
from utils.custom import css_code  # Ensure this file exists or remove this line if unnecessary

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Progress bar function
def progress_bar(amount_of_time: int) -> Any:
    progress_text = "Please wait, Generative models hard at work"
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(amount_of_time):
        time.sleep(0.04)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

# Function to generate text from an image using Hugging Face's image captioning model
def generate_text_from_image(url: str) -> str:
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    generated_text = image_to_text(url)[0]["generated_text"]
    print(f"IMAGE INPUT: {url}")
    print(f"GENERATED TEXT OUTPUT: {generated_text}")
    return generated_text

# Function to generate a story from text using ChatOpenAI
def generate_story_from_text(scenario: str) -> str:
    prompt_template = """
    You are a talented storyteller who can create a story from a simple narrative.
    Create a story using the following scenario; the story should be a maximum of 50 words long.

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["scenario"])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)
    generated_story = story_llm.predict(scenario=scenario)
    print(f"TEXT INPUT: {scenario}")
    print(f"GENERATED STORY OUTPUT: {generated_story}")
    return generated_story

# Function to generate speech from text using Hugging Face's ESPnet TTS model
def generate_speech_from_text(message: str) -> Any:
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payloads = {"inputs": message}
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open("generated_audio.flac", "wb") as file:
        file.write(response.content)

# Main function to run the Streamlit app
def main() -> None:
    st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼️")
    st.markdown(css_code, unsafe_allow_html=True)  # Custom CSS if needed

    with st.sidebar:
        st.image("img/gkj.jpg")
        st.write("---")
        st.write("AI App created by @ Tech DareDevils")

    st.header("Image-to-Story Converter")
    uploaded_file = st.file_uploader("Please choose a file to upload", type="jpg")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Display progress
        progress_bar(100)
        
        # Generate text and story
        scenario = generate_text_from_image(uploaded_file.name)
        story = generate_story_from_text(scenario)
        
        # Generate audio
        generate_speech_from_text(story)

        # Display results
        with st.expander("Generated Image Scenario"):
            st.write(scenario)
        with st.expander("Generated Short Story"):
            st.write(story)

        # Play generated audio
        st.audio("generated_audio.flac")

# Run the main function
if __name__ == "__main__":
    main()
