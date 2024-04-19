SEMANTIC_IMPORTANCE = 0.8  # Value 0-1
# Tweak the above variable to toggle between semantic importance and empathy

SEMANTIC_IMPORTANCE_FOR_RESPONSE_COMPARISON = 0.7  # Value 0-1
ROUGE_IMPORTANCE_FOR_RESPONSE_COMPARISON = 0.2  # Value 0-1
BLEU_IMPORTANCE_FOR_RESPONSE_COMPARISON = 0.1  # Value 0-1

CONTEXT_SOURCE_SCORE_THRESHOLD = 0.75

AI_IMPORTANCE_FOR_EVAL = 0.5
HUMAN_IMPORTANCE_FOR_EVAL = 0.1
RESPONSE_IMPORTANCE_FOR_EVAL = 0.4

def compute_ai_response_score(**kwargs):
    # Step 1: Declaration
    if kwargs.get('is_for_context'):
        empathy = kwargs.get('ai_context_empathy')
        similarity = kwargs.get('ai_context_similarity')
    else:
        empathy = kwargs.get('ai_source_empathy')
        similarity = kwargs.get('ai_source_similarity')
    
    # Step 2: Return a normalised score
    score = SEMANTIC_IMPORTANCE * similarity + (1 - SEMANTIC_IMPORTANCE) * empathy

    # Step 3: Return the score
    return score

def compute_human_response_score(**kwargs):
    # Step 1: Declaration
    if kwargs.get('is_for_context'):
        empathy = kwargs.get('human_context_empathy')
        similarity = kwargs.get('human_context_similarity')
    else:
        empathy = kwargs.get('human_source_empathy')
        similarity = kwargs.get('human_source_similarity')
    
    # Step 2: Return a normalised score
    score = SEMANTIC_IMPORTANCE * similarity + (1 - SEMANTIC_IMPORTANCE) * empathy

    # Step 3: Return the score
    return score

def compute_response_comparison_score(**kwargs):
    # Step 1: Decalarations
    bleu = kwargs.get('responses_bleu')
    rouge = kwargs.get('responses_rouge')
    similarity = kwargs.get('responses_similarity')

    # Step 2: Processing
    rouge_values = list(rouge.values())
    rouge = (0.2 * rouge_values[0]) + (0.2 * rouge_values[1]) + (0.6 * rouge_values[2])

    score = (SEMANTIC_IMPORTANCE_FOR_RESPONSE_COMPARISON * similarity) + (ROUGE_IMPORTANCE_FOR_RESPONSE_COMPARISON * rouge) + (BLEU_IMPORTANCE_FOR_RESPONSE_COMPARISON * bleu)

    # Return the score
    return score

def compute_eval_score_for_response(**kwargs):
    # Step 1: Compute all scores
    # Get computed score for ai response w.r.t source conversation
    ai_source_score = compute_ai_response_score(**kwargs) 

    # Get computed score for human response w.r.t source conversation
    human_source_score = compute_human_response_score(**kwargs)


    kwargs['is_for_context'] = True  
    # Get computed score for ai response w.r.t context/current conversation 
    ai_context_score = compute_ai_response_score(**kwargs)

    # Get computed score for human response w.r.t context/current conversation 
    human_context_score = compute_human_response_score(**kwargs)

    # Get ai-human response comparison score
    response_comparison_score = compute_response_comparison_score(**kwargs)

    # Get source-context similarity
    context_source_score = kwargs.get('source_context_similarity')

    # incase conversations are similar > 0.75 use the source scores as well
    if context_source_score > CONTEXT_SOURCE_SCORE_THRESHOLD:
        normalizer = 1 + context_source_score
        ai_score = (ai_context_score + (context_source_score * ai_source_score)) / normalizer
        human_score = (human_context_score + (context_source_score * human_source_score)) / normalizer
    else:
        ai_score = ai_context_score
        human_score = human_context_score
    
    # Step 3:
    score = (AI_IMPORTANCE_FOR_EVAL * ai_score) + (HUMAN_IMPORTANCE_FOR_EVAL * human_score) + (RESPONSE_IMPORTANCE_FOR_EVAL * response_comparison_score)

    # Step 4: Return score
    return score
    