import os
import sys
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu

sys.path.append(os.path.abspath('../../'))
from src.metrics.compute_cosine import compute_similarity
from src.metrics.compute_empathy import measure_empathy
from src.metrics.compute_rouge import compute_rouge_scores
from src.data_handling.utils import compute_eval_score_for_response

model = SentenceTransformer('BAAI/bge-base-en-v1.5')


def score_data_point(data_point):
    """
    Given a data point computes all relevant scores

    Args:
        - prev_context_conversation (list)
        - source_conversation (list)
        - ai_response (str)
        - human_response (str)
    
    Returns:
        - scores (dict): Holds all computed scores
        - score (float)
    """
    scores = {}
    prev = data_point['prev_context_conversation']
    prev_conversation_text = ''
    for msg in prev:
        prev_conversation_text += f' {list(msg.values())[0]}'

    # Concatenate source conversation messages into a single string
    source = data_point['source_conversation']
    source_conversation_text = ''
    for msg in source:
        source_conversation_text += f' {list(msg.values())[0]}'

    # AI and human responses
    ai_response = data_point['ai_response']
    human_response = data_point['human_response']

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

    return scores, score
