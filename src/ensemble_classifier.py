import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LANG_DIR = DATA_DIR / "language"
TRAIN_DIR = LANG_DIR / "training_data"
MODEL_DIR = ROOT / "models" / "prose_models"

from encorporate import Encorporator
"""
dataframe structure:
                dataframe({
                #text metrics:
                "text": text, 
                "pos": [t.upper() for w, t in tagged_tokens], 
                "lemmas": lemmas,
                #document level metrics:
                "mean sentence length": mean_sentence,
                "std sentence length": std_sentence,
                "lexical density": lexical_density,
                "variance": variance,
                "burstiness": burstiness,
                "saliency": sum(frequencies)/len(frequencies) if frequencies else 0,
                "sentiment": aggregate_sentiment,
                "sentiment deviation": std_sentiment,
                "ttr": TTR,
                #syntax tree data (averages/maxes/% for all sentences):
                "depth": max_depth, 
                "head": mean_head,
                "sub clauses": mean_subclauses, 
                "coord clauses": mean_coordconjs, 
                "branching": mean_branchbias, 
                "balanced": pct_balanced,
                "label": label,
                "timestamp": timestamp
                })
"""
metrics = ["mean sentence length", "std sentence length", "lexical density", "variance", "burstiness", "saliency", "sentiment", "sentiment deviation", "ttr", 
           "depth", "head", "sub clauses", "coord clauses", "branching", "balanced"]

language = ["text", "pos", "lemmas"]

human_df = pd.read_csv(TRAIN_DIR / "human_prose_data.csv")
ai_df = pd.read_csv(TRAIN_DIR / "llm_prose_data.csv")

full_df = pd.concat([human_df, ai_df], ignore_index=True)

for text_col in ["text", "pos", "lemmas"]:
    full_df[text_col] = full_df[text_col].fillna("").astype(str)


def stringify_lists(df, columns=["pos", "lemmas"]):
    for col in columns:
        df[col] = df[col].astype(str).str.replace(r"[\[\]',]", "", regex=True)
    
    return df

df = stringify_lists(df=full_df)



X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

text_pre = ColumnTransformer(
        transformers=[
            ('txt_body', TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), 'text'),
            ('txt_pos', TfidfVectorizer(max_features=1000, ngram_range=(2, 4)), 'pos'),
            ('txt_lem', TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), 'lemmas')
        ]
)

text_pipeline = Pipeline(steps=[
    ('preprocessor', text_pre),
    ('svm', SVC(kernel='linear', probability=True, random_state=42))
    ]
)


metrics_pre = ColumnTransformer(
        transformers=[
            ('num_scaler', StandardScaler(), metrics)
        ]
)

metrics_pipeline = Pipeline(steps=[
    ('preprocessor', metrics_pre),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42))
    ]
)


ensemble = VotingClassifier(
        estimators=[
            ('text_expert', text_pipeline),
            ('metrics_expert', metrics_pipeline)
        ],
        voting='soft'
)

print("Training ensemble model on text and metrics...")
ensemble.fit(X_train, y_train)

print("Evaluation: ")
predictions = ensemble.predict(X_test)
print(classification_report(y_test, predictions))

parser = Encorporator()

human_text_sample = """
    The categorization of crime seems at first to be a classical category of the “necessary and sufficient” variety, or even more likely a legal category. But as with most things in language, a closer analysis reveals that there is much more nuance to the topic. Crime can be considered a radial category with murder as a central member, and many layers of “less criminal” and “more ambiguous" radiating out from that center. 
	If we take murder to be the most central and prototypical example of crime, we can likely include theft, assault, and rape directly alongside it. But as we move away from the most criminal, and the most literal uses of the term, we discover less exemplary members of the crime family. Moving just outside of the most severe, we can place things like civil infractions such as parking tickets pretty far away from the most prototypical. When a person says they got a parking ticket we hardly consider them a criminal, or even the action a crime, in the most prototypical sense. There are also many examples of victimless crimes, which contains in the name the sense that we consider it less criminal than those crimes with clear victims. 
	At the farthest periphery of the category we can find things like bad fashion choices or unreasonably priced goods.  It is common to hear a bad deal described as “highway robbery,” and while this may be less an example of the deal itself being categorized as crime and more a case of a metaphorical extension, it can be argued that we collectively view the behavior of charging more than is necessary or reasonable as criminal. As crimes go, however, this falls about as far away as possible from robbing a liquor store. 
	Crime as a radial category exemplifies how prototypes exist in language, and furthermore how we expand and extend terms to include many concepts or items that would not originally have been described by that word.
    """

ai_text_sample = """
    In cognitive linguistics, a radial category is a conceptual structure where member terms are organized around a central, prototypical instance, with less representative members branching outward via metaphorical, metonymic, or experiential extensions. When applied to the sociology of law, the concept of "crime" functions precisely as a radial category rather than a classical set defined by rigid, necessary, and sufficient conditions. At the prototypical core of the category lies the widely recognized "street crime"—acts of direct, physical harm or theft, such as homicide or bank robbery. This central archetype forms the cognitive baseline for what society intuitively labels as criminal, characterized by clear intent, immediate harm, and an identifiable victim.

    Radiating outward from this cognitive core are non-prototypical crimes that diverge from the archetype while retaining a structural link to the concept of lawbreaking. White-collar crimes, such as embezzlement, insider trading, or corporate fraud, sit on these peripheral rays. While they lack the immediate, physical violence of the prototype, they are classified as crimes because they share the underlying abstract features of deception, systemic harm, and illegal financial gain. Similarly, victimless crimes like illicit gambling or drug possession stretch the category further, removing the element of a direct, non-consenting victim while maintaining the violation of state-mandated legal frameworks.

    By analyzing crime through this radial framework, it becomes evident that the boundaries of legality are fluid and culturally contingent, rather than binary. The periphery of the category is constantly shifting to accommodate evolving societal values and technological advancements, as seen in the emergence of cybercrimes or corporate environmental negligence. These marginal examples challenge traditional legal definitions, but their connection to the prototypical core is maintained through extended legal and moral reasoning. Ultimately, viewing crime as a radial category highlights how human conceptualization relies on prototypes and flexible extensions to navigate complex social and ethical landscapes.
    """

test_human_df = parser.frame([human_text_sample])
test_ai_df = parser.frame([ai_text_sample])

for textcol in ["text", "pos", "lemmas"]:
    test_human_df[textcol] = test_human_df[textcol].fillna("").astype(str)
    test_ai_df[textcol] = test_ai_df[textcol].fillna("").astype(str)
                
test_human_df = stringify_lists(test_human_df)
test_ai_df = stringify_lists(test_ai_df)

unseen_human_X = test_human_df[X_train.columns]
unseen_ai_X = test_ai_df[X_train.columns]

unseen_samples = [
        ("Expected HUMAN", unseen_human_X),
        ("Expected LLM", unseen_ai_X)
        ]

for description, data in unseen_samples:
    prediction = ensemble.predict(data)[0]
    P = ensemble.predict_proba(data)[0]
    human_score = P[0] * 100
    ai_score = P[1] * 100 

    print(f"\nTesting: {description}")
    print(f"Classifier predicts: {prediction}")
    print(f"Confidence: human: {human_score}, ai: {ai_score}")



model_file = MODEL_DIR / "paired_data_ensemble_model.joblib"
print(f"Saving trained model to {str(model_file)}")
joblib.dump(ensemble, model_file)




