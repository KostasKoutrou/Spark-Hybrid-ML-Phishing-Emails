# Spark-Hybrid-ML-Phishing-Emails
A Python program to test combinations of types of features and classifiers using Spark

SpyderTest.py runs all the combinations of classifiers with simple features and the hybrid "assembled" combinations, which use both properties of the emails and text-based features.

stackmodel.py tests a Hybrid "stacked" model, where there are two classifiers used:
The first one is trained using text-based features (either Word Embedding or TF-IDF)
The second one is outputs the final out of the architecture. It is trained using properties-based features and an additional feature which is the classification of the first classifier for the to-be-classified email.
