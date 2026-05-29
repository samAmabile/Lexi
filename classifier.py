from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
import joblib 
import nltk

from pathlib import Path

from encorporate import Encorporator, Codecorpus


"""
dataframe structure: ({

                "text": text, 
                "pos": [t.upper() for w, t in tagged_tokens], 
                "lemmas": lemmas,
                "mean sentence length": mean_sentence,
                "std sentence length": std_sentence,
                "lexical density": lexical_density,
                "variance": variance,
                "burstiness": burstiness,
                "saliency": sum(frequencies)/len(frequencies) if frequencies else 0,
                "sentiment": aggregate_sentiment,
                "sentiment deviation": std_sentiment,
                "ttr": TTR,
                #syntax tree data - calculate averages/maxes for all sentences in doc:
                "depth": max_depth, 
                "head": mean_head,
                "sub clauses": mean_subclauses, 
                "coord clauses": mean_coordconjs, 
                "branching": mean_branchbias, 
                "balanced": sum_balance,
                "label": label,
                "timestamp": timestamp

            })
"""

class ai_prose_detector:
    
    def __init__(self):
        self.ling_data = ["text", "pos", "lemmas"]
        self.metrics = ["mean sentence length", "std sentence length", "lexical density", "variance", "burstiness", "saliency", "sentiment", "sentiment deviation", "depth", "head", "sub clauses", "coord clauses", "branching", "balanced"]
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        self.model_file = None
        self.vectorizers_file = None
        self.vectorizers = {}
        self.sent_vec = None
        self.pos_vec = None
        self.lemma_vec = None

    def get_metadata(self, df):
        metadata_df = df[self.metrics].copy()

        if "balanced" in metadata_df.columns:
            metadata_df["balanced"] = metadata_df["balanced"].astype(int)

        return metadata_df

    def get_lingdata(self, df): 

        self.sent_vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)
        sent_x = self.sent_vec.fit_transform((df["sentence"]))

        self.pos_vec = CountVectorizer(analyzer='word', ngram_range=(1, 3))
        pos_x = self.pos_vec.fit_transform(df["pos"].apply(lambda x: ' '.join(x) if isinstance(x, list) else x))

        self.lemma_vec = CountVectorizer(analyzer='word', max_features=3000)
        lemma_x = self.lemma_vec.fit_transform(df["lemmas"].apply(lambda x: ' '.join(x) if isinstance(x, list) else x))

        self.vectorizers = {
                "sent": self.sent_vec, 
                "pos": self.pos_vec, 
                "lemma": self.lemma_vec, 
                "metrics": self.metrics
                }

        return sent_x, pos_x, lemma_x

    def vectorize_data(self, metadata_df, sent_x, pos_x, lemma_x, labels=None):
        meta_x = csr_matrix(metadata_df.values)
 
        X = hstack([meta_x, sent_x, pos_x, lemma_x])

        return X, labels

    def train_classifier(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        #model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)

        self.model.fit(X_train, Y_train)

        prediction = self.model.predict(X_test)

        accuracy = accuracy_score(Y_test, prediction)
        
        return accuracy

    def save_training(self, save_as=""):
        if not save_as:
            save_as = "llm_classifier"

        model_filename = f"{save_as}_model.joblib"
        vectorizers_filename = f"{save_as}_vectorizers.joblib"

        path = "classifiers"

        Path(path).mkdir(parents=True, exist_ok=True)

        model_path = Path(path) / model_filename
        vectorizers_path = Path(path) / vectorizers_filename

        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizers, vectorizers_path)

        return model_filename, vectorizers_filename

    def load_model(self, model_filename, vectorizers_filename):
        self.model = joblib.load(model_filename)
        self.vectorizers = joblib.load(vectorizers_filename)
        self.sent_vec = self.vectorizers["sent"]
        self.pos_vec = self.vectorizers["pos"]
        self.lemma_vec = self.vectorizers["lemma"]

        return self.model, self.vectorizers

    def format_unseen(self, text):

        parser = Encorporator()
        parsed_text = parser.frame(text, False)

        return parsed_text

    def vectorize_files(self, filename_a, filename_b):

        df_a = pd.read_csv(filename_a)
        df_b = pd.read_csv(filename_b)

        df = pd.concat([df_a, df_b], ignore_index=True)


        df["sentence"] = (
                df["sentence"]
                .astype(str)
                .str.replace(" .", ".", regex=False)
                .str.replace(" ,", ",", regex=False)
                .str.replace(" ?", "?", regex=False)
                .str.replace(" !", "!", regex=False)
                .str.replace("``", '"', regex=False)
                .str.replace("''", '"', regex=False)
                .str.replace("#", "", regex=False)
                .str.replace("*", "", regex=False)
                )

        df = df[df["sentence"].astype(str).str.strip().str.len() > 3]

        doc_df = df.groupby(['timestamp', 'label']).agg({
            "sentence": lambda x: ' '.join(str(val) for val in x),
            "pos": lambda x: ' '.join(' '.join(pos) if isinstance(pos, list) else str(pos) for pos in x),
            "lemmas": lambda x: ' '.join(' '.join(lemma) if isinstance(lemma, list) else str(lemma) for lemma in x),
            "lexical density": "mean",
            "variance": "mean",
            "burstiness": "mean", 
            "saliency": "mean",
            "sentiment": "mean",
            "sentiment deviation": "mean", 
            "depth": "mean",
            "head": "mean",
            "sub clauses": "mean",
            "coord clauses": "mean",
            "branching": "mean",
            "balanced": "mean"
            }).reset_index()

        Y = doc_df["label"].map({"LLM": 0, "HUMAN": 1}).fillna(1).astype(int).values

        metadata = self.get_metadata(doc_df)
        sent_vec, pos_vec, lemma_vec = self.get_lingdata(doc_df)
        X, Y = self.vectorize_data(metadata, sent_vec, pos_vec, lemma_vec, labels=Y)

        return X, Y

    ####
    def fit_model(self, llm_file, human_file):

        X, Y = self.vectorize_files(llm_file, human_file)
        accuracy = self.train_classifier(X, Y)

        self.model_file, self.vectorizers_file = self.save_training()

        return accuracy

    def vectorize_unseen(self, df):
        
        if "balanced" in df.columns:
            df["balanced"] = df["balanced"].astype(int)


        
        sent_vec = self.vectorizers['sent']
        pos_vec = self.vectorizers['pos']
        lemma_vec = self.vectorizers['lemma']
        metrics = self.vectorizers['metrics']

        document = {
                "sentence": " ".join(df["sentence"].astype(str)),
                "pos": " ".join(df["pos"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))),
                "lemmas": " ".join(df["lemmas"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)))
                }

        for metric in metrics:
            if metric in df.columns:
                document[metric] = df[metric].mean()
            else:
                document[metric] = 0.0

        document_df = pd.DataFrame([document])


        sent_x = sent_vec.transform(document_df["sentence"])
        pos_x = pos_vec.transform(document_df["pos"])
        lemma_x = lemma_vec.transform(document_df["lemmas"])
        meta_x = csr_matrix(document_df[metrics].values)

        X = hstack([meta_x, sent_x, pos_x, lemma_x])
   
        return X

    def classify(self, text):

        X = self.vectorize_unseen(self.format_unseen(text))

        classification = self.model.predict(X)[0]

        label = "HUMAN" if classification == 1 else "LLM"

        return label


if __name__ == "__main__":

    llm_file = "llm_language/master.csv"
    human_file = "natural_language/master.csv"

    classifier = ai_prose_detector()

    accuracy = classifier.fit_model(llm_file, human_file)

    ai_text = """
    The Scotch bonnet pepper (Capsicum chinense) boasts a rich, centuries-old history deeply intertwined with the transatlantic slave trade and the cultural shaping of the Caribbean. Native to Central and South America, the wild ancestors of these fiery peppers were cultivated by the indigenous Taíno and Arawak peoples long before European arrival. Following European colonization, the pepper was integrated into the forced migration routes of the Middle Passage. Enslaved Africans, already familiar with hot peppers in their homelands, quickly adopted the Scotch bonnet, cultivating it in small provision grounds. Over generations, it became a foundational crop, adapting to the tropical Caribbean climate and embedding itself as a vital agricultural staple.

    The pepper owes its distinct, whimsical name to its physical appearance, which closely resembles a traditional Scottish Tam o' Shanter bonnet hat. Featuring a squat, crumpled shape with three to four distinct lobes, it was named by British colonists who noticed the visual similarity to the flat, woolen caps worn by Scottish laborers and soldiers. While the name reflects a European colonial perspective, the pepper itself became the undeniable soul of West Indian identity. As regional cooking styles evolved, the Scotch bonnet transitioned from a basic garden crop to the culinary signature of Jamaica and neighboring islands, symbolizing resilience and cultural synthesis.

    Today, the Scotch bonnet is celebrated globally, famous not just for its intense heat—ranging from 100,000 to 350,000 Scoville Heat Units—but for its unique, sweet, and fruity flavor profile. It serves as the irreplaceable backbone of traditional Caribbean cuisine, driving the complex flavor of Jamaican jerk seasonings, Trinidadian pepper sauces, and West African stews. Beyond its culinary impact, the pepper holds economic significance as a major export crop for Caribbean nations, sustaining local farming communities. From its indigenous roots and colonial naming to its modern status as a gourmet staple, the history of the Scotch bonnet reflects a journey of survival, adaptation, and global culinary triumph.
    """
    human_text = """
    The categorization of crime seems at first to be a classical category of the "necessary and sufficient" variety, or even more likely a legal category. But as with most things in language, a closer analysis reveals that there is much more nuance to the topic. Crime can be considered a radial category with murder as a central member, and many layers of "less criminal" and "more ambiguous" radiating out from that center. 
	If we take murder to be the most central and prototypical example of crime, we can likely include theft, assault, and rape directly alongside it. But as we move away from the most criminal, and the most literal uses of the term, we discover less exemplary members of the crime family. Moving just outside of the most severe, we can place things like civil infractions such as parking tickets pretty far away from the most prototypical. When a person says they got a parking ticket we hardly consider them a criminal, or even the action a crime, in the most prototypical sense. There are also many examples of victimless crimes, which contains in the name the sense that we consider it less criminal than those crimes with clear victims. 
	At the farthest periphery of the category we can find things like bad fashion choices or unreasonably priced goods.  It is common to hear a bad deal described as “highway robbery,” and while this may be less an example of the deal itself being categorized as crime and more a case of a metaphorical extension, it can be argued that we collectively view the behavior of charging more than is necessary or reasonable as criminal. As crimes go, however, this falls about as far away as possible from robbing a liquor store. 
	Crime as a radial category exemplifies how prototypes exist in language, and furthermore how we expand and extend terms to include many concepts or items that would not originally have been described by that word. 
    """

    model_prediction_1 = classifier.classify(ai_text)
    model_prediction_2 = classifier.classify(human_text)

    print(f"On unseen data, model predicts '{model_prediction_1}' for a sample of ai generated text and '{model_prediction_2}' for a sample of human generated text")
    print(f"trained model accuracy: {accuracy}")






        




















    

























