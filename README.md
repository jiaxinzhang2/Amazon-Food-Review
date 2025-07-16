# Amazon-Food-Review
Use NLP techniques to extract sentiment (positive, negative, neutral) from user comments on a new Amazon product feature for food-related items. The goal is to determine whether users respond favorably to the feature and whether it should be expanded.


1. Processed over 500K Amazon food reviews using Pandas and NLTK, applying stopword removal, stemming, and case normalization for robust text preprocessing.
2. Extracted features using unigrams, bigrams, and TF-IDF via TfidfVectorizer to convert text into structured inputs for model training.
3. Trained and evaluated Logistic Regression and Random Forest classifiers for binary sentiment analysis with the metrics including AUC, F1-score, and Precision/Recall.
4. Visualized feature importance to interpret predictive phrases and enhance model explainability.
