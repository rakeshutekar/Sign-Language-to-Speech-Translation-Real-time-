import os

import openai
from dotenv import load_dotenv
from openai.openai_object import OpenAIObject

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def load_class_names():
    """
    Load class index to label mapping from a file.
    """
    idx2label = {}
    with open('wlasl_class_list.txt', 'r') as f:
        for line in f:
            idx2label[int(line.split()[0])]=line.split()[1]
    return idx2label


def update_word_buffer(word_buffer, new_word):
    """
    Updates the word buffer with the new word, maintaining a fixed size.
    """
    word_buffer.append(new_word)
    if len(word_buffer) > 5:  # Keep the buffer size to the last 5 words
        word_buffer.pop(0)
    return word_buffer


def predict_next_word_with_llm(word_buffer):
    """
    Use the LLM (e.g., GPT-3.5 Turbo or GPT-4 Turbo) to predict the next word based on the context of the word_buffer.
    Instruct the model explicitly not to end its response with a question mark.
    """
    prompt = " ".join(word_buffer) + " "
    instruction = " Predict the next word without ending your response with a question mark."
    full_prompt = prompt + instruction

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4-turbo-preview" based on your preference
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ]
    )
    
    # Extracting the model's response
    if isinstance(response, OpenAIObject):
        response_text = response.get('choices')[0].get('message').get('content').strip()
    else:
        response_text = response['choices'][0]['message']['content'].strip()

    next_word = response_text.split()[-1]  # Assuming the response is a continuation and we take the last word as prediction
    
    # Ensure the next word does not end with a question mark, just in case
    next_word = next_word.rstrip('?')

    return next_word
