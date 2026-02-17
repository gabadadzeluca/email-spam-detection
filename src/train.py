from src.feature_extraction import extract_features
from sklearn.naive_bayes import MultinomialNB

# separate labels and data
def prepare_training_data(texts):
  labels = [text[0] for text in texts]
  data = [text[1] for text in texts]
  return labels, data

def train_model(texts):
  labels, data = prepare_training_data(texts)
  features, tfidf_vectorizer = extract_features(data)
  model = MultinomialNB()
  model.fit(features, labels)
  return model, tfidf_vectorizer