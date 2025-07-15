import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utils import (
    load_gemini_pro_model,
    gemini_pro_vision_response,
    text_embeddings_response,
    gemini_response,
)

working_dir = os.path.dirname(os.path.abspath(__file__))
# print(working_dir)


st.set_page_config(
    page_title="Chat with Gemini",
    page_icon="🧠",
    layout="centered",
    # initial_sidebar_state="expanded",
    # menu_items={
    #     "Get Help": "https://www.extremelycoolapp.com/help",
    #     "Report a bug": "https://www.extremelycoolapp.com/bug",
    #     "About": "# This is a header. This is an *extremely* cool app!",
    # },
)


with st.sidebar:

    selected = option_menu(
        "Gemini Chat Menu",
        [
            "Gemini ChatBot",
            "Image Captioning",
            "Embed Text",
            "Ask me Anything",
        ],
        menu_icon="robot",
        icons=["chat-dots-fill", "image-fill", "textarea-t", "patch-question-fill"],
        default_index=0,
    )


def translate_role_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role


if selected == "Gemini ChatBot":
    model = load_gemini_pro_model()

    # init sessions
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    st.title("🤖 ChatBot")

    # chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    user_prompt = st.chat_input("Ask Gemini...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        # print(st.chat_message("user").markdown(user_prompt))
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)


if selected == "Image Captioning":
    st.title("📷 Snap Narrate")

    user_image = st.file_uploader(
        "Upload an Image to generate caption...", type=["jpg", "jpeg", "png", "svg"]
    )

    if st.button("Generate Caption"):
        if user_image:
            img = Image.open(user_image)
            col1, col2 = st.columns(2)
            with col1:
                resized_img = img.resize((800, 500))
                st.image(resized_img)

            default_prompt = "Write a short caption for this image"
            caption = gemini_pro_vision_response(default_prompt, img)

            with col2:
                st.info(caption)
        else:
            st.info("Please upload an Image and try again")


if selected == "Embed Text":
    st.title("🔢 Embed Text")

    user_prompt = st.text_input("Enter text to embed...")

    if st.button("Get Embeddings"):
        if user_prompt:
            output = text_embeddings_response(user_prompt)
            st.text(output)
        else:
            st.info("Please upload some Text and try again")


if selected == "Ask me Anything":
    st.title("🤔 Ask Gemini...")

    user_prompt = st.text_input("Ask Gemini anything...")

    if st.button("Ask"):
        if user_prompt:
            output = gemini_response(user_prompt)
            st.markdown(output)
        else:
            st.info("Please enter a prompt and try again")
