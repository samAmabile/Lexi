import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

text_features = ["text", "pos", "lemmas"]

llm_df = pd.read_csv("llm_language/master.csv")
hmn_df = pd.read_csv("natural_language/master.csv")

df = pd.concat([llm_df, hmn_df], ignore_index=True)

for col in ["pos", "lemmas"]:
    df[col] = df[col].astype(str).str.replace(r"[\[\]',]", "", regex=True)

df["sentence"] = (
        df["sentence"]
        .astype(str)
        .str.replace(" . . ", ". ", regex=False)
        .str.replace(" , , ", ", ", regex=False)
        .str.replace(" ? ? ", "? ", regex=False)
        .str.replace(" ! ! ", "! ", regex=False)
        .str.replace(" ? ", "? ", regex=False)
        .str.replace(" ! ", "! ", regex=False)
        .str.replace("``", '"', regex=False)
        .str.replace("''", '"', regex=False)
        .str.replace('""', '"', regex=False)
        .str.replace("#", "", regex=False)
        .str.replace("*", "", regex=False)
)

document_df = df.groupby(['timestamp', 'label']).agg({
    'sentence': lambda x: " ".join(x), 
    'pos': lambda x: " ".join(x),
    'lemmas': lambda x: " ".join(x),
    'lexical density': 'mean', 
    'variance': 'mean',
    'burstiness': 'mean',
    'saliency': 'mean',
    'depth': 'mean',
    'sentiment deviation': 'mean', 
    'document ttr': 'first',
    'balanced': 'sum'
}).reset_index()

X = document_df.drop(columns=["timestamp", "label"])
y = document_df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
num_features = ['lexical density', 'variance', 'burstiness', 'saliency', 'depth', 'sentiment deviation', 'document ttr']


preprocessor = ColumnTransformer(
        transformers=[
            ('sent_txt', TfidfVectorizer(max_features=5000), 'sentence'), 
            ('pos_txt', TfidfVectorizer(max_features=1000), 'pos'),
            ('lemma_txt', TfidfVectorizer(max_features=3000), 'lemmas'),
            ('num', StandardScaler(), num_features)
        ]
)

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('classifier', SVC(kernel='linear'))
])

svm_pipeline.fit(X_train, y_train)
predictions = svm_pipeline.predict(X_test)
print(classification_report(y_test, predictions))





