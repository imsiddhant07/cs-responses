from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_rouge_scoress(data_list):
    """method to compute rouge scores

    Args:
        data_list (list): data from csv file

    Returns:
    """
    # Step 1: Declarations
    scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []},
    }

    # Step 2: Iterate over data_list to compute scores
    for data in data_list:
        score = scorer.score(data[2], data[3])
        for k, v in score.items():
            scores[k]['precision'].append(v.precision)
            scores[k]['recall'].append(v.recall)
            scores[k]['fmeasure'].append(v.fmeasure)
    
    for k,v in scores.items():
        scores[k]['precision'] = sum(scores[k]['precision']) / len(scores[k]['precision'])
        scores[k]['recall'] = sum(scores[k]['recall']) / len(scores[k]['recall'])
        scores[k]['fmeasure'] = sum(scores[k]['fmeasure']) / len(scores[k]['fmeasure'])
        
    return scores

def compute_rouge_scores(source, target):
    """method to compute rouge scores

    Args:
        data_list (list): data from csv file

    Returns:
    """
    score = scorer.score(source, target)
    return score