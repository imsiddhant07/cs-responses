from nltk.translate.bleu_score import sentence_bleu

# def compute_bleu(data_list):
#     """method to compute bleu scores

#     Args:
#         data_list (list): data from csv file

#     Returns:
#         - unigram_score (float)
#         - bigram_score (float)
#         - trigram_score (float)
#         - quadgram_score (float)
#         - generalised_score (float)
#     """
#     # Step 1: Declarations
#     unigram_scores = []
#     bigram_scores = []
#     trigram_scores = []
#     quadgram_scores = []
#     generalised_scores = []

#     # Step 2: Iterate over the data_list
#     for data in data_list:
#         # weights here represent importance to n-gram
#         # weight (1, 0, 0, 0) - importance only to uni-gram
#         # weight (0, 0, 1, 0) - importance only to tri-gram
#         unigram_scores.append(sentence_bleu([data[2]], data[3], weights=(1, 0, 0, 0)))
#         bigram_scores.append(sentence_bleu([data[2]], data[3], weights=(0, 1, 0, 0)))
#         trigram_scores.append(sentence_bleu([data[2]], data[3], weights=(0, 0, 1, 0)))
#         quadgram_scores.append(sentence_bleu([data[2]], data[3], weights=(0, 0, 0, 1)))
#         generalised_scores.append(sentence_bleu([data[2]], data[3]))
    
#     # Step 3: Compute the n-gram scores
#     unigram_score = sum(unigram_scores) / len(unigram_scores)
#     bigram_score = sum(bigram_scores) / len(bigram_scores)
#     trigram_score = sum(trigram_scores) / len(trigram_scores)
#     quadgram_score = sum(quadgram_scores) / len(quadgram_scores)
#     generalised_score = sum(generalised_scores) / len(generalised_scores)

#     # Step 4: Return the n-gram scores
#     return (unigram_score, bigram_score, trigram_score, quadgram_score, generalised_score)

bleu_weight_mapping = {
    'all': (0.25, 0.25, 0.25, 0.25),
    'uni': (1, 0, 0, 0),
    'bi': (0, 1,  0, 0),
    'tri': (0, 0, 1, 0),
    'quad': (0, 0, 0, 1),
}

def compute_bleu_score(source, target, n_gram='all'):
    """method to compute bleu scores

    Args:
        - source (str): string for comparison
        - target (str): string for comparison
        - n_gram (str): string for generating n-gram score
            - all 
            - uni
            - bi
            - tri
            - quad

    Returns:
        - score (float)
    """
    # Step 1: Compute bleu score based on n-grams
    score = sentence_bleu([source], target, weights=bleu_weight_mapping[n_gram])

    # Step 2: Return the score
    return score
