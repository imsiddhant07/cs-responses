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
    """
    Calculate a normalized score for an AI's response based on its empathy and similarity.

    This function takes into account whether the score is being computed in relation
    to the context of the conversation or the source material. The score is a weighted
    sum of empathy and semantic similarity, with the ability to assign more importance
    to either component through a SEMANTIC_IMPORTANCE factor.

    Args:
    **kwargs : dict
        Variable keyword arguments that can include:
        - is_for_context (bool): Flag to determine if the score is for the context of the conversation.
        - ai_context_empathy (float): Empathy score for the AI's response in the context of the conversation.
        - ai_context_similarity (float): Semantic similarity score for the AI's response in the context of the conversation.
        - ai_source_empathy (float): Empathy score for the AI's response in relation to the source material.
        - ai_source_similarity (float): Semantic similarity score for the AI's response in relation to the source material.

    Returns:
        - float : The normalized score for the AI's response, which is a weighted average of the 
            empathy and semantic similarity scores based on the SEMANTIC_IMPORTANCE factor.

    Notes:
        - The SEMANTIC_IMPORTANCE is a predefined constant that determines the relative
        weight of the similarity score in the final score calculation.
        - The empathy component is weighted as (1 - SEMANTIC_IMPORTANCE).
        - If 'is_for_context' is True, the function uses 'ai_context_empathy' and 'ai_context_similarity';
        otherwise, it uses 'ai_source_empathy' and 'ai_source_similarity'.
    """
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
    """Calculate a normalized score for a human's response based on empathy and semantic similarity.

    This function computes a score reflecting the quality of a human's response in a conversation,
    considering two aspects: empathy and similarity. It accounts for the response in relation
    to the context of the conversation or the source material, based on the provided keyword
    arguments. The score is a weighted combination of empathy and semantic similarity, with the
    relative importance of similarity determined by the constant SEMANTIC_IMPORTANCE.

    Args:
        **kwargs: A dictionary of keyword arguments that can include:
            - is_for_context (bool): A flag that indicates whether the score should be calculated
                against the context of the conversation (True) or the source material (False).
            - human_context_empathy (float): The empathy score of the human's response with respect
                to the context of the conversation. Required if is_for_context is True.
            - human_context_similarity (float): The semantic similarity score of the human's response
                with respect to the context of the conversation. Required if is_for_context is True.
            - human_source_empathy (float): The empathy score of the human's response with respect
                to the source material. Required if is_for_context is False.
            - human_source_similarity (float): The semantic similarity score of the human's response
                with respect to the source material. Required if is_for_context is False.

    Returns:
        A float representing the normalized score of the human's response, factoring in both
        empathy and similarity, weighted by SEMANTIC_IMPORTANCE.

    Notes:
        - SEMANTIC_IMPORTANCE is a predefined constant that should be set prior to calling this
          function, reflecting the importance of semantic similarity in the final score.
        - The empathy score is weighted by (1 - SEMANTIC_IMPORTANCE).
        - It is assumed that the appropriate empathy and similarity scores are passed in through
          the keyword arguments; otherwise, the function may not behave as expected.

    """
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
    """Computes an evaluation score for response comparison based on BLEU, ROUGE, and semantic similarity.

    This score is a weighted sum of the BLEU score, ROUGE score, and semantic similarity, where each
    component's importance is defined by predefined constants.

    Args:
        - responses_bleu (float): The BLEU score comparing AI and human responses.
        - responses_rouge (dict): A dictionary of ROUGE scores for AI and human response comparison.
        - responses_similarity (float): The semantic similarity score between AI and human responses.

    Returns:
        - The final evaluation score as a float, calculated by weighting the BLEU, ROUGE, and similarity
        scores according to their predefined importance.
    """
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
    """
    Calculates an overall evaluation score for AI and human responses based on multiple criteria.

    This function integrates scores for both AI and human responses with respect to the source
    material and the context of the conversation. It also includes a comparison score between AI
    and human responses. If the source and context are significantly similar, both scores are
    adjusted. The final score is a weighted average of these components.

    Args:
        **kwargs: Keyword arguments that include scores for empathy and similarity, and
                  a flag to indicate context relevance. Expected keys are:
            - ai_source_empathy (float)
            - ai_source_similarity (float)
            - human_source_empathy (float)
            - human_source_similarity (float)
            - ai_context_empathy (float)
            - ai_context_similarity (float)
            - human_context_empathy (float)
            - human_context_similarity (float)
            - responses_comparison_score (float)
            - source_context_similarity (float)

    Returns:
        The final evaluation score as a float.
    """
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
    
def compute_ai_response_score(**kwargs):
    """
    Calculates the evaluation score for an AI response considering its relevance to both the
    source material and the context of the conversation.

    This function computes two scores for the AI response: one with respect to the source
    material and the other with respect to the current context. It then normalizes these scores
    based on the similarity between the source and context. If the source and context are
    sufficiently similar, the source score is taken into account; otherwise, only the context
    score is used.

    Args:
        **kwargs: A dictionary of keyword arguments that includes:
            - ai_source_empathy (float): The empathy score of the AI's response to the source.
            - ai_source_similarity (float): The similarity score of the AI's response to the source.
            - ai_context_empathy (float): The empathy score of the AI's response to the context.
            - ai_context_similarity (float): The similarity score of the AI's response to the context.
            - source_context_similarity (float): The similarity score between the source and the context.

    Returns:
        The normalized score for the AI's response as a float.
    """
    # Step 1: Compute all scores
    # Get computed score for ai response w.r.t source conversation
    ai_source_score = compute_ai_response_score(**kwargs) 

    kwargs['is_for_context'] = True  
    # Get computed score for ai response w.r.t context/current conversation 
    ai_context_score = compute_ai_response_score(**kwargs)

    # Get source-context similarity
    context_source_score = kwargs.get('source_context_similarity')

    # incase conversations are similar > 0.75 use the source scores as well
    if context_source_score > CONTEXT_SOURCE_SCORE_THRESHOLD:
        normalizer = 1 + context_source_score
        ai_score = (ai_context_score + (context_source_score * ai_source_score)) / normalizer
    else:
        ai_score = ai_context_score
    
    # Step 3:
    score = ai_score

    # Step 4: Return score
    return score

def compute_human_response_score(**kwargs):
    """
    Calculates a score for a human response based on its relevance to both source material and the context.

    This function assesses how well a human's response aligns with the source conversation and the context
    of the current conversation. If the source and context are similar beyond a certain threshold, the
    source score is factored into the final score; otherwise, only the context score is considered.

    Args:
        **kwargs: Keyword arguments that include empathy and similarity scores, and a context flag. Expected keys:
            - human_source_empathy (float): Empathy score for the source conversation.
            - human_source_similarity (float): Semantic similarity score for the source conversation.
            - human_context_empathy (float): Empathy score for the context conversation.
            - human_context_similarity (float): Semantic similarity score for the context conversation.
            - source_context_similarity (float): Similarity between the source and the context conversations.

    Returns:
        A normalized score (float) for the human response, considering both source and context relevance.

    """
    # Step 1: Compute all scores
    # Get computed score for human response w.r.t source conversation
    human_source_score = compute_human_response_score(**kwargs) 

    kwargs['is_for_context'] = True  
    # Get computed score for human response w.r.t context/current conversation 
    human_context_score = compute_human_response_score(**kwargs)

    # Get source-context similarity
    context_source_score = kwargs.get('source_context_similarity')

    # incase conversations are similar > 0.75 use the source scores as well
    if context_source_score > CONTEXT_SOURCE_SCORE_THRESHOLD:
        normalizer = 1 + context_source_score
        human_score = (human_context_score + (context_source_score * human_source_score)) / normalizer
    else:
        human_score = human_context_score
    
    # Step 3:
    score = human_score

    # Step 4: Return score
    return score