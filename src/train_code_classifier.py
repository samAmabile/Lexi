import pandas as pd
import joblib
import numpy as np
import random 
from encorporate import Codecorpus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
TRAIN_DIR = DATA_DIR / "code" / "train"

def find_best_metrics(X_train_scaled, y_train):
    paramater_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }
    grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), paramater_grid, cv=5, scoring='f1_macro')
    grid.fit(X_train_scaled, y_train)

    return grid.best_params_, grid.best_score_


def calculate_variance(sourcecode):
    lines = str(sourcecode).splitlines()

    if len(lines) < 3: 
        return 0.0

    indentations = []
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            indentations.append(indent)

    if len(indentations) < 3:
        return 0.0

    variance = np.var(indentations)

    return variance

def calculate_blanklines(sourcecode):
    intervals = []
    cur_gap = 0
    codelines = sourcecode.splitlines()
    for line in codelines:
        if not line.strip():
            cur_gap += 1
        else:
            intervals.append(cur_gap)
            cur_gap = 0

    if len(intervals) < 2:
        return 0.0

    mean = np.mean(intervals)
    std = np.std(intervals)

    CV = float(std/mean)

    return CV

def normalize_metrics(df): 
    df["token density"] = df["no. tokens"] / df["loc"]
    df["node density"] = df["no. nodes"] / df["no. tokens"]
    df["function density"] = df["no. functions"] / df["loc"]
    df["node ratio"] = df["no. nodes"] / df["loc"]
    df["burstiness"] = df["complexity"] / df["loc"]
    df["variance"] = df["src"].apply(calculate_variance)
    df["gap variance"] = df["src"].apply(calculate_blanklines)

    df.replace([np.inf, -np.inf, np.nan], 0.0, inplace=True)


models_folder = Path("code_models")
models_folder.mkdir(parents=True, exist_ok=True)

ai_df = pd.read_csv(TRAIN_DIR / "ai_train_14k.csv")
hum_df = pd.read_csv(TRAIN_DIR / "human_train_16k.csv")



def remove_firstline(text): 
    return '\n'.join(text.splitlines()[1:])

ai_df["src"] = ai_df["src"].apply(lambda x: remove_firstline(x))
ai_df["label"] = 1
hum_df["label"] = 0
#normalize_metrics(ai_df)
#normalize_metrics(hum_df)

for df in [ai_df, hum_df]:
    df["is_cpp"] = (df["lang"] == "cpp").astype(float)
    df["lang"] = df["lang"].replace({"c": "c_cpp", "cpp": "c_cpp"})

combined_df = pd.concat([ai_df, hum_df], ignore_index=True) 

##Naive Bayes Language detection: 

nb_model = MultinomialNB(alpha=1.0)

X_trn, X_tst, y_trn, y_tst = train_test_split(
        combined_df["src"] ,
        combined_df["lang"],
        test_size=0.2, 
        random_state=42
)

nb_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]*\b', max_features=5000, min_df=3)

Xtrn_tfidf = nb_vectorizer.fit_transform(X_trn)
Xtst_tfidf = nb_vectorizer.transform(X_tst)

nb_model.fit(Xtrn_tfidf, y_trn)
y_pred = nb_model.predict(Xtst_tfidf)
print("\nMultinomial Bayes Programming Language Classifier - Training Report: ")
print(classification_report(y_tst, y_pred))

nb_vec_file = MODEL_DIR / models_folder / "vectorizerNB.joblib"
lang_router = MODEL_DIR / models_folder / "language_router.joblib"

joblib.dump(nb_vectorizer, nb_vec_file)
joblib.dump(nb_model, lang_router)


##Start SVC classifier here : 

    #df columns: src,lang,loc,no. tokens,no. nodes,no. functions,avg variable length,max depth,control density,complexity,label,timestamp#

df_dict_ai = {label: frame for label, frame in ai_df.groupby("lang")}
df_dict_humn = {label: frame for label, frame in hum_df.groupby("lang")}

master_df_dict = {}
metrics_by_lang = {}


metrics = ["ttr", "avg variable length", "max depth","nested depth", "control density", "complexity", "token density", "node density", "node ratio", "function density", "blankline ratio", "blankline variance", "blankspace ratio", "blankspace variance", "burstiness", "shannon entropy", "linter score", "comment density", "is_cpp", "label" ]


for lang, aidf in df_dict_ai.items():
    if lang in df_dict_humn:
        humndf = df_dict_humn[lang]

        master_df_dict[lang] = pd.concat([aidf, humndf], ignore_index=True)
        metrics_by_lang[lang] = master_df_dict[lang][metrics]
    else:
        master_df_dict[lang] = aidf
        metrics_by_lang[lang] = master_df_dict[lang][metrics]

svm_exprts = {}
lang_scalers = {}

for language, df in metrics_by_lang.items():
    print(f"\nTraining {language.upper()} expert for ai detection")

    X = df.drop(columns=["label"])
    y = df["label"]

    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_trn_scaled = scaler.fit_transform(X_trn)
    X_tst_scaled = scaler.transform(X_tst)

    lang_scalers[language] = scaler

    clf = SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale")
    clf.fit(X_trn_scaled, y_trn)
    
    svm_exprts[language] = clf

    y_prd = clf.predict(X_tst_scaled)

    best_params, best_score = find_best_metrics(X_trn_scaled, y_trn)

    print(f"Best {language.upper()} paramaters: {best_params} | Best Score: {best_score}")

    print(classification_report(y_tst, y_prd))

#save models:
for lang in svm_exprts.keys():

    model_save = MODEL_DIR / models_folder / f"{lang}_expert.joblib"
    scaler_save = MODEL_DIR / models_folder / f"{lang}_scaler.joblib"

    joblib.dump(svm_exprts[lang], model_save)
    joblib.dump(lang_scalers[lang], scaler_save)



print(f"All detection models saved to {models_folder.resolve()}")




    









