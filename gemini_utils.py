import os
import json
import google.generativeai as genai
from PIL import Image

working_dir = os.path.dirname(os.path.abspath(__file__))
# print(working_dir)

config_file_path = f"{working_dir}\\config.json"
config_data = json.load(open(config_file_path))


GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)


def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("models/gemini-2.0-flash")
    return gemini_pro_model


def gemini_pro_vision_response(prompt, image):
    gemini_pro_vision_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_pro_vision_model.generate_content([prompt, image])
    result = response.text
    print(result)
    return result


def text_embeddings_response(input_text):
    embedding_model = "models/embedding-001"
    embeddings = genai.embed_content(
        model=embedding_model,
        content=input_text,
        task_type="retrieval_document",
    )
    return embeddings


def gemini_response(prompt):
    gemini_pro_model = genai.GenerativeModel("models/gemini-2.0-flash")
    result = gemini_pro_model.generate_content(prompt)
    # print(gemini_response("global crisis-es"))
    return result.text






# test runs
#
# image = Image.open('test_image.png')
# prompt = "Write a short caption for thi image"
# output = gemini_pro_vision_response(prompt, image)
# print(output)
#
# output = text_embeddings_response("Hello python streamlit")
# print(output)
