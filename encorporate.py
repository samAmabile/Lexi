#ANALYSIS MODULE FOR PROSE AND CODE
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize 
from nltk import word_tokenize 
from nltk.sentiment import SentimentIntensityAnalyzer

#from tree_sitter_language_pack import get_parser
import tree_sitter_python as tspy
import tree_sitter_cpp as tscpp
import tree_sitter_javascript as tsjs
import tree_sitter_html as tshtml
import tree_sitter_java as tsjava
import tree_sitter_c as tsc
from tree_sitter import Language, Parser
import lizard
import spacy

import pandas as pd
import re
from pathlib import Path
from datetime import datetime
import numpy as np


#lemmatizer = WordNetLemmatizer()
#sia = SentimentIntensityAnalyzer()

def get_pos(tag): 
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('vader_lexicon')

#stop_words = set(stopwords.words('english'))

LLMLANG = "llm_language"
NLANG = "natural_language"
LLMCODE = "llm_code"
HCODE = "human_code"

class Encorporator: 
    def __init__(self, text=""): 
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_sm")
        
        for folder in [LLMLANG, NLANG]:
            Path(folder).mkdir(parents=True, exist_ok=True)
    def encorporate(self, text, llm=True): 
        dataset = []
        sentences = sent_tokenize(text)
        all_tokens = word_tokenize(text)
        TTR = len(set(all_tokens)) / len(all_tokens) if len(all_tokens) > 0 else 0
        fdist, n = self.freq_dist(text)
        aggregate_sentiment = self.analyze_sentiment(text)['compound']
        for sent in sentences: 

            sentiment = self.analyze_sentiment(sent)['compound']
            
            syntax_tree = self.analyze_syntax(sent)
            
            max_depth = syntax_tree["depth"]
            head_location = syntax_tree["head"]
            sub_clauses = syntax_tree["sub_clauses"]
            coordng_conjs = syntax_tree["coord_conjs"]
            branch_bias = syntax_tree["branching"]
            balanced = syntax_tree["balanced"]


            tokens = word_tokenize(sent)
            tagged_tokens = nltk.pos_tag(tokens)

            lexical_density = self.lexical_density(tagged_tokens)

            lemmas = []

            for word, tag in tagged_tokens:
                wn_pos = get_pos(tag)
                lemma = self.lemmatizer.lemmatize(word, wn_pos)
                lemmas.append(lemma)
            
            frequencies = [fdist[w]/n for w in set(tokens)] if n > 0 else 0
            variance = float(np.var(frequencies))
            mean_freq = float(np.mean(frequencies))
            freq_sd = float(np.std(frequencies))

            burstiness = freq_sd / mean_freq if mean_freq != 0 else 0
            
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

            label = "LLM" if llm else "HUMAN"
            dataset.append({
                "sentence": sent, 
                "pos": [t.upper() for w, t in tagged_tokens], 
                "lemmas": lemmas,
                "lexical density": lexical_density,
                "variance": variance,
                "burstiness": burstiness,
                "saliency": sum(frequencies)/len(frequencies) if frequencies else 0,
                "sentiment": sentiment,
                "sentiment deviation": abs(sentiment - aggregate_sentiment),
                "document ttr": TTR,
                "depth": max_depth, 
                "head": head_location,
                "sub clauses": sub_clauses, 
                "coord clauses": coordng_conjs, 
                "branching": branch_bias, 
                "balanced": balanced,
                "label": label,
                "timestamp": timestamp
                })
        
        topical_text = " ".join(text.split()[:20])
        title = self.generate_title(topical_text)
        date = datetime.now().strftime('%Y-%m-%d_%H%M')
        filename = f"{title}_{date}.csv"
        path = self.save_dataframe(dataset, filename, llm)
        return str(path)
    
    def lexical_density(self, tagged_tokens):
        if not tagged_tokens:
            return 0

        tags = {
            'NN', 'NNS', 'NNP', 'NNPS',
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
            'JJ', 'JJR', 'JJS', 
            'RB', 'RBR', 'RBS'
            }

        relevant_words = [w for w, t in tagged_tokens if t in tags]

        lex_density = len(relevant_words) / len(tagged_tokens)

        return lex_density
    
    def analyze_syntax(self, sentence):
        doc = self.nlp(sentence)
        sent = list(doc.sents)[0]
        root = sent.root

        def dfs(node):
            if not list(node.children):
                return 1
            return 1 + max(dfs(child) for child in node.children)

        max_depth = dfs(root)
        head = root.i / len(sent)
        
        subordinate_clauses = sum(1 for word in sent if word.dep_ in ('advcl', 'ccomp', 'relcl'))
        coordinating_conjunctions = sum(1 for word in sent if word.dep_ == 'cc')

        left_nodes = sum(len(list(token.lefts)) for token in sent)
        right_nodes = sum(len(list(token.rights)) for token in sent) 

        bias = (right_nodes - left_nodes) / len(sent)

        metrics = {
                "depth": max_depth, 
                "head": head, 
                "sub_clauses": subordinate_clauses, 
                "coord_conjs": coordinating_conjunctions, 
                "branching": bias, 
                "balanced": abs(left_nodes - right_nodes) <= 2
                }

        return metrics

    def generate_title(self, text):

        if not text:
            return datetime.now().strftime('%Y-%m-%d_%H%M')

        if len(text.split()) < 6:
            words = text.split()
            text = " ".join([w for w in words if w.lower() not in self.stop_words])
            text = re.sub(r'[^\w\s-]', '', text.lower()).replace(' ', '_')
            if not text: 
                return datetime.now().strftime('%Y-%m-%d_%H%M')
            return text

        words = re.findall(r'\w+', text.lower())

        tagged = nltk.pos_tag([w for w in words if w not in self.stop_words])
        lemmas = [self.lemmatizer.lemmatize(w, get_pos(t)) for w, t in tagged]

        fdist = nltk.FreqDist(lemmas)

        title_set = set([w for w, count in fdist.most_common(5)])

        title = "_".join([w for w in words if w in title_set])

        return title

    def save_dataframe(self, dataset, filename, llm=True):
        folder = LLMLANG if llm else NLANG
        #Path(folder).mkdir(parents=True, exist_ok=True)
        filename = filename+".csv" if not filename.endswith(".csv") else filename

        df = pd.DataFrame(dataset)
        path = Path(folder) / filename
        master = Path(folder) / "master.csv"

        header_bool = master.exists()

        df.to_csv(path, index=False)
        df.to_csv(master, mode='a', index=False, header=not header_bool)

        return path

    def freq_dist(self, text): 
        tokens = [w.lower() for w in re.findall(r'\w+', text) if w not in self.stop_words]

        fdist = nltk.FreqDist(tokens)
        num_words = len(tokens)

        return fdist, num_words
    
    def analyze_sentiment(self, text):
        
        sentiment = self.sia.polarity_scores(text)

        return sentiment

class Codecorpus:
    def __init__(self, code=""): 
        self.langmap = {
                "py": "python", "python": "python", "ipynb": "python",
                "cpp": "cpp", "js": "javascript", "java": "java", 
                "c": "c"
                }
        self.langtrees = {
                "py" : Language(tspy.language()), 
                "python": Language(tspy.language()),
                "cpp": Language(tscpp.language()),
                "c": Language(tsc.language()), 
                "js": Language(tsjs.language()),
                "javascript": Language(tsjs.language()), 
                "java": Language(tsjava.language()),
                "html": Language(tshtml.language())
                }
        for folder in [LLMCODE, HCODE]:
            Path(folder).mkdir(parents=True, exist_ok=True)
    def analyze(self, codebase, lang, llm=True):

        label = "LLM" if llm else "HUMAN"
        dataset = []

        parser_lang = self.langtrees[lang.strip().lower()]
        parser = Parser(parser_lang)

        for code in codebase:

            complexity, lines, num_tokens = self.metrics(code, lang)


            tree = parser.parse(bytes(code, 'utf-8'))
            tree_stats = self.ast_stats(tree, lines)

            num_nodes = tree_stats["nodes"]
            max_depth = tree_stats["depth"]
            avg_var_len = tree_stats["avg_var_len"]
            control_density = tree_stats["control_density"]
            num_functions = tree_stats["functions"]

            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
            label = "LLM" if llm else "HUMAN"

            analysis = {
                    "src": code,
                    "loc": lines, 
                    "no. tokens": num_tokens, 
                    "no. nodes": num_nodes, 
                    "no. functions": num_functions, 
                    "avg variable length": avg_var_len, 
                    "max depth": max_depth,
                    "control density": control_density, 
                    "complexity": complexity,
                    "label": label,
                    "timestamp": timestamp
                    }
            dataset.append(analysis)
        
        savetime = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        filename = f"{label}_{lang}_{savetime}.csv"
        path = self.save_dataframe(dataset, filename, llm)

        return str(path)


    def analyze_file(self, code, lang, llm=True):

        label = "LLM" if llm else "HUMAN"
        dataset = []

        parser_lang = self.langtrees[lang.strip().lower()]
        parser = Parser(parser_lang)

        complexity, lines, num_tokens = self.metrics(code, lang)

        tree = parser.parse(bytes(code, 'utf-8'))
        tree_stats = self.ast_stats(tree, lines)

        num_nodes = tree_stats["nodes"]
        max_depth = tree_stats["depth"]
        avg_var_len = tree_stats["avg_var_len"]
        control_density = tree_stats["control_density"]
        num_functions = tree_stats["functions"]

        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        label = "LLM" if llm else "HUMAN"

        analysis = {
                "src": code,
                "loc": lines, 
                "no. tokens": num_tokens, 
                "no. nodes": num_nodes, 
                "no. functions": num_functions, 
                "avg variable length": avg_var_len, 
                "max depth": max_depth,
                "control density": control_density, 
                "complexity": complexity,
                "label": label,
                "timestamp": timestamp
                }
        dataset.append(analysis)
        
        savetime = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        filename = f"{label}_{lang}_{savetime}.csv"
        path = self.save_dataframe(dataset, filename, llm)

        return str(path)
    
    def ast_stats(self, tree, LoC):
        stats = {
            "nodes": 0, 
            "depth": 0, 
            "nest_depth": 0,
            "var_lengths": [], 
            "avg_var_len": 0,
            "control_blocks": 0, 
            "control_density": 0,
            "functions": 0
            }

        def dfs(node, depth, nested_depth):
            stats["nodes"] += 1
            stats["depth"] = max(stats["depth"], depth)

            cur_nesting = nested_depth + 1 if any(x in node.type for x in ["for_statement", "while_statement", "do_statement"]) else nested_depth
            stats["nest_depth"] = max(stats["nest_depth"], cur_nesting)

            if node.type == "identifier":

                parent_type = node.parent.type if node.parent else ""
                valid_parents = ["declarator", "assignment", "paramater", "for_in", "pointer", "field", "variable_declarator", "assignment_expression", "paramater_declaration"]

                if any(p in parent_type for p in valid_parents):
                    name = node.text.decode('utf-8', errors='ignore')
                    length = len(name)
                    stats["var_lengths"].append(length)

            if any(x in node.type for x in ["if_statement", "for_statement", "while", "while_statement", "case"]):
                stats["control_blocks"] += 1

            function_ids = {"function_definition", "method_definition", "method_declaration", "function_item", "constructor_declaration"}
            
            if node.type in function_ids:
                stats["functions"] += 1
                   
            for child in node.children:
                dfs(child, depth+1, cur_nesting)
        
        root = tree.root_node if hasattr(tree, "root_node") else tree
        dfs(root, 1, 0)

        avg_var_len = sum(stats["var_lengths"]) / len(stats["var_lengths"]) if stats["var_lengths"] else 0
        ctrl_dnsty = stats["control_blocks"] / LoC if LoC != 0 else 0

        stats["avg_var_len"] = avg_var_len
        stats["control_density"] = ctrl_dnsty

        return stats

    def metrics(self, code, lang): 

        analysis = lizard.analyze_file.analyze_source_code(f"snippet.{lang}", code)

        avg_cyclomatic = analysis.average_cyclomatic_complexity
        loc = analysis.nloc
        num_tokens = analysis.token_count

        return avg_cyclomatic, loc, num_tokens

    
    def save_dataframe(self, dataset, filename, llm=True):
        folder = LLMCODE if llm else HCODE
        Path(folder).mkdir(parents=True, exist_ok=True)
        filename = filename+".csv" if not filename.endswith(".csv") else filename

        df = pd.DataFrame(dataset)
        path = Path(folder) / filename
        master = Path(folder) / "master.csv"

        header_bool = master.exists()

        df.to_csv(path, index=False)
        df.to_csv(master, mode='a', index=False, header=not header_bool)

        return path









            

        

        

