from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from explore_data import read_data

texts = read_data()[0] # Get training data only for feature extraction

def extract_features(texts):
  """
    # For my understanding:
    # ignore less then min_df=5 documents
    # ignore words that appear in more then max_df=0.8 (ratio) documents
    # use sublinear tf scaling, i.e. replace tf with 1 + log(tf)
    # use idf weighting
  """

  tfidf = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
  features = tfidf.fit_transform([text[1] for text in texts])
  df = pd.DataFrame(
    features.toarray(), 
    columns=tfidf.get_feature_names_out()
  )
  print("Extracted features shape:", df.shape)

  # Look at index-th email's non-zero features
  doc_index = 0
  non_zero = df.iloc[doc_index][df.iloc[doc_index] > 0]
  print(non_zero)

  return features, tfidf