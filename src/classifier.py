from encorporate import Encorporator
import pandas as pd
import numpy as np 
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models" / "prose_models"
LANG_DIR = DATA_DIR / "language"
TEST_DIR = LANG_DIR / "tests"


class Classifier:

    def __init__(self):
        self.model_bin = MODEL_DIR / "paired_data_ensemble_model.joblib"
        self.model = joblib.load(self.model_bin)
        self.text_model = self.model.named_estimators_['text_expert']
        self.metrics_model = self.model.named_estimators_['metrics_expert']
        self.metrics = ["mean sentence length", "std sentence length", "lexical density", "variance", 
           "burstiness", "saliency", "sentiment", "sentiment deviation", "ttr", 
           "depth", "head", "sub clauses", "coord clauses", "branching", "balanced"]
        self.language = ["text", "pos", "lemmas"]
        self.features = self.language + self.metrics
        self.parser = Encorporator()
        
    
    def stringify_columns(self, df, columns=["pos", "lemmas"]):
        for col in columns:
            df[col] = df[col].astype(str).str.replace(r"[\[\]',]", "", regex=True)
        
        return df

    def frame_text(self, text, label=False):
        df = self.parser.frame(text, label)
        for col in self.language:
            df[col] = df[col].fillna("").astype(str)

        df = self.stringify_columns(df)

        return df
    def classify_metrics(self, text):
        df = self.frame_text([text])

        X = df[self.features]
        prediction = self.metrics_model.predict(X)[0]
        P = self.metrics_model.predict_proba(X)[0]

        label = "HUMAN" if prediction == 0 else "LLM"

        #print(f"text is {label} written")
        #print(f"likely HUMAN: {P[0]*100:.2f}% | likely LLM: {P[1]*100:.2f}%")

        return label

    def classify_text(self, text):

        df = self.frame_text([text])

        X = df[self.features]
        prediction = self.text_model.predict(X)[0]
        P = self.text_model.predict_proba(X)[0]

        label = "HUMAN" if prediction == 0 else "LLM"

        #print(f"text is {label} written")
        #print(f"likely HUMAN: {P[0]*100:.2f}% | likely LLM: {P[1]*100:.2f}%")

        return label

    def classify(self, text):

        df = self.frame_text([text])

        X = df[self.features]
        prediction = self.model.predict(X)[0]
        P = self.model.predict_proba(X)[0]

        print(f"text is {prediction} written")
        print(f"Likelihood HUMAN: {P[0] * 100:.2f}% | Likelihood LLM: {P[1] * 100:.2f}%")

        return prediction

    def interface(self):
        while True:
            text = input("Enter text to classify (enter 'X' to exit): ")
            if text.strip().lower() == 'x':
                break
            if text.endswith(".txt"):
                text = open(text, 'r', encoding='utf-8').read()

            self.classify(text)

if __name__ == "__main__":
    human_str = ""
    ai_str = ""
    
    with open(TEST_DIR / "human_written_tests.txt", 'r') as hum, open(TEST_DIR / "ai_written_tests.txt", 'r') as ai:
        human_str = hum.read().strip()
        ai_str = ai.read().strip()

    human_docs = human_str.split('#END#')

    ai_docs = ai_str.split('#END#')

    classifier = Classifier()

    predictions = []
    text_predictions = []
    metrics_predictions = []
    num_human = len(human_docs)
    num_ai = len(ai_docs)

    true_labels = [1]*num_human + [0]*num_ai

    for doc in human_docs:
        predictions.append(classifier.classify(doc)) 
    
    for doc in ai_docs:
        predictions.append(classifier.classify(doc))

    labels = [0 if x == 'LLM' else 1 for x in predictions]

    print(predictions)
    print(labels)
    pct_human = sum(labels) / len(labels)
    pct_ai = 1 - pct_human

    actual_human = sum(true_labels) / len(true_labels)
    actual_ai = 1 - actual_human
    
    print(f"model guessed ai for {pct_ai*100:.2f}% of texts, actual is {(num_ai/len(labels))*100:.2f}%")

    print(f"model guessed human for {pct_human*100:.2f}% of texts, actual is {(num_human/len(labels))*100:.2f}%")

    print(f"Actual ai documents = {actual_ai*100:.2f}%")
    print(f"Actual human documents = {actual_human*100:.2f}%")










    

