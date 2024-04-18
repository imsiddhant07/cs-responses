from nltk.sentiment import SentimentIntensityAnalyzer

def measure_empathy(response):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(response)
    empathy_score = scores['compound']
    return empathy_score
