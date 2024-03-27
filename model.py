import os

import cv2
import playsound
import torch
from gtts import gTTS

from pytorch_i3d import InceptionI3d
from utils import predict_next_word_with_llm, update_word_buffer


def load_model(dataset='WLASL2000'):
    """
    Load the InceptionI3d model with specified dataset configuration.
    """
    to_load = {
        'WLASL2000':{'logits':2000,'path':'weights/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'}
    }

    model = InceptionI3d(num_classes=400)
    model.load_state_dict(torch.load('weights/rgb_imagenet.pt', map_location='cpu'))
    model.replace_logits(to_load[dataset]['logits'])

    if 'path' in to_load[dataset]:
        model.load_state_dict(torch.load(to_load[dataset]['path'], map_location='cpu'), strict=False)

    model.eval()
    model.cpu()
    return model

def classify_live(input_tensor, model, idx2label, word_buffer, start_ignore=12, threshold=0.5):
    """
    Classify the live input tensor, deciding whether to use the WLASL model's output directly
    or enhance it with the LLM based on confidence scores after ignoring the first 2 words.
    """
    with torch.no_grad():
        per_frame_logits = model(input_tensor)

    predictions = torch.mean(per_frame_logits, dim=2)
    _, top_indices = torch.topk(predictions, 1)  # Just need the top prediction for this logic
    top_index = top_indices.cpu().numpy()[0][0]

    predictions = torch.nn.functional.softmax(predictions, dim=1).cpu().numpy()[0]
    top_prediction = idx2label[top_index]
    top_prediction_confidence = predictions[top_index]

    # Initialize a static variable to keep track of the number of words processed
    if "word_count" not in classify_live.__dict__:
        classify_live.word_count = 0

    if classify_live.word_count < start_ignore or top_prediction_confidence >= threshold:
        word_to_speak = top_prediction
    else:
        # Use the LLM to predict the next word based on the recent history
        word_to_speak = predict_next_word_with_llm(word_buffer)
        print(f"LLM Prediction: {word_to_speak}")  # For debugging

    # If not the initial words, update the word buffer with the chosen word
    if classify_live.word_count >= start_ignore:
        update_word_buffer(word_buffer, word_to_speak)

    # Convert the chosen word to speech
    tts = gTTS(text=word_to_speak, lang='en')
    tts_file = 'prediction.mp3'
    tts.save(tts_file)
    playsound.playsound(tts_file)
    os.remove(tts_file)

    classify_live.word_count += 1  # Increment the word count after processing

    return word_to_speak  # Return the word as a string
