"""
cd src/data_handling
python score_builder.py
"""

import sys
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu

sys.path.append(os.path.abspath('../../'))
from src.metrics.compute_cosine import compute_similarity
from src.metrics.compute_empathy import measure_empathy
from src.metrics.compute_rouge import compute_rouge_scores
from src.data_handling.utils import compute_eval_score_for_response

data_path = '../../data'


# Step 1: Read data from the structured JSON files for equal data
with open(f'{data_path}/structured_equal.json', 'r') as f:
    equal_data = json.load(f)  # Load the data from JSON file into a Python list

# Step 2: Read data from the structured JSON files for non-equal data
with open(f'{data_path}/structured_non_equal.json', 'r') as f:
    non_equal_data = json.load(f)

# Step 3: Load model for computing embeddings
model = SentenceTransformer('BAAI/bge-base-en-v1.5')


"""
prev_conversation_text <> source_conversation_text - cosine
C-AI : 
  - prev_conversation_text + ai_response - empathy
  - prev_conversation_text <> ai_response - cosine
C-H : 
  - prev_conversation_text + human_response - empathy
  - prev_conversation_text <> human_response - cosine
S-AI : 
  - source_conversation_text + ai_response - empathy
  - source_conversation_text <> ai_response - cosine
S-H : 
  - source_conversation_text + human_response - empathy
  - source_conversation_text <> human_response - cosine
AI-H :
    - ai_response <> human_response - cosine
    - ai_response <> human_response - bleu (1 values)
    - ai_response <> human_response - rouge (3 values)
"""



def compute_scores_for_bulk_data(data_obj):
    for idx in tqdm(range(len(data_obj)-30, len(data_obj))):
        scores = {}
        # Concatenate previous context messages into a single string
        prev = data_obj[idx]['prev_context_conversation']
        prev_conversation_text = ''
        for msg in prev:
            prev_conversation_text += f' {list(msg.values())[0]}'

        # Concatenate source conversation messages into a single string
        source = data_obj[idx]['source_conversation']
        source_conversation_text = ''
        for msg in source:
            source_conversation_text += f' {list(msg.values())[0]}'

        # AI and human responses
        ai_response = data_obj[idx]['ai_response']
        human_response = data_obj[idx]['human_response']

        # Compute cosine similarity between source and previous context texts
        source_context_similarity = compute_similarity(model, source_conversation_text, prev_conversation_text)
        scores['source_context_similarity'] = float(source_context_similarity)

        # Compute empathy and similarity for AI's response with the previous context
        ai_prev_conversation_text = prev_conversation_text + ai_response
        ai_context_empathy = measure_empathy(ai_prev_conversation_text)
        scores['ai_context_empathy'] = float(ai_context_empathy)

        ai_context_similarity = compute_similarity(model, prev_conversation_text, ai_response)
        scores['ai_context_similarity'] = float(ai_context_similarity)

        # Compute empathy and similarity for Human's response with the previous context
        human_prev_conversation_text = prev_conversation_text + human_response
        human_context_empathy = measure_empathy(human_prev_conversation_text)
        scores['human_context_empathy'] = float(human_context_empathy)

        human_context_similarity = compute_similarity(model, prev_conversation_text, human_response)
        scores['human_context_similarity'] = float(human_context_similarity)

        # Compute empathy and similarity for AI's response with the source context
        ai_source_conversation_text = source_conversation_text + ai_response
        ai_source_empathy = measure_empathy(ai_source_conversation_text)
        scores['ai_source_empathy'] = float(ai_source_empathy)

        ai_source_similarity = compute_similarity(model, source_conversation_text, ai_response)
        scores['ai_source_similarity'] = float(ai_source_similarity)

        # Compute empathy and similarity for Human's response with the source context
        human_source_conversation_text = source_conversation_text + human_response
        human_source_empathy = measure_empathy(human_source_conversation_text)
        scores['human_source_empathy'] = float(human_source_empathy)

        human_source_similarity = compute_similarity(model, source_conversation_text, human_response)
        scores['human_source_similarity'] = float(human_source_similarity)

        # Compute similarity, ROUGE scores, and BLEU score between AI's and Human's responses
        responses_similarity = compute_similarity(model, ai_response, human_response)
        scores['responses_similarity'] = float(responses_similarity)

        responses_rouge = compute_rouge_scores(ai_response, human_response)
        scores['responses_rouge'] = {}
        scores['responses_rouge']['rouge1'] = float(responses_rouge['rouge1'].fmeasure)
        scores['responses_rouge']['rouge2'] = float(responses_rouge['rouge2'].fmeasure)
        scores['responses_rouge']['rougeL'] = float(responses_rouge['rougeL'].fmeasure)

        responses_bleu = sentence_bleu([ai_response], human_response, weights=(0.25,0.25,0.25,0.25))
        scores['responses_bleu'] = float(responses_bleu)

        score = compute_eval_score_for_response(**scores)

        data_obj[idx]['scores'] = scores
        data_obj[idx]['score'] = score

    return data_obj


data_obj = compute_scores_for_bulk_data(equal_data)
with open(f'{data_path}/scored_equal.json', 'w') as f:
    json.dump(data_obj, f, indent=4)

data_obj = compute_scores_for_bulk_data(non_equal_data)
with open(f'{data_path}/scored_non_equal.json', 'w') as f:
    json.dump(data_obj, f, indent=4)
