from gemini import Automate, Code
from encorporate import Encorporator, Codecorpus

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import os
import random 
import time
from typing import List, cast
from datetime import datetime
import csv
import sys

import nltk 
from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.corpus import webtext
from nltk import sent_tokenize

from transformers import AutoTokenizer
from magika import Magika



class dataset_builder_ai:

    def __init__(self, api_key="GEMINI_API_KEY"):

        api_var = ""
        if api_key in os.environ:
            api_var = os.environ.get(api_key)
        else:
            api_var = api_key

        self.API = api_var

        self.prose_generator = Automate(self.API)
        self.code_generator = Code(self.API)

        self.prose_parser = Encorporator()
        self.code_parser = Codecorpus()
        self.topicfile = [line for line in open("bottopics.txt", 'r')]

    def generate_corpus(self, topic, calls=6):
        chatname = self.prose_generator.automate_loop(topic, calls)

        print(f"Chat log saved to {chatname}")
        
        rawtexts = []
        if chatname is not None:
            df = pd.read_csv(chatname) 
            llm_content = (df["Content"].dropna().astype(str).tolist())
            rawtexts = llm_content
        else:
            print("Failed to retrieve content from model")
            return None

        corpusname = self.prose_parser.encorporate(rawtexts, True)

        if corpusname is not None: 
            print(f"annotated chat saved to {corpusname}")
            return corpusname
        else:
            print("Failed to annotate chat")
            return None

    def generate_code_corpus(self, prompt, iterations=6, language=None): 
        codename, codefile = self.code_generator.automate_code(prompt, iterations, language)

        print(f"Code file saved to {codename}")

        if codename is not None: 
            df = pd.read_csv(codename, usecols=["response", "language"])
            filenames = []
            for lang, group in df.groupby("language"):
                codebase = group["response"].tolist()

                langstr = str(lang).strip().lower()

                print(f"processing {len(codebase)} codefiles in language {langstr}")

                csv = self.code_parser.analyze(codebase, langstr, llm=True)

                print(f"annotated codebase saved to {csv}")

                filenames.append(csv)

            if len(filenames) == 1:
                return str(filenames[0])
            elif len(filenames) > 1: 
                return filenames
            else:
                print("failed to annotate codebase")
                return None
        else:
            print("failed to generate code")
            return None

    def generate_codebase(self, language, promptfile=None, calls=6): 
        
        csvpath = self.code_generator.generate_code_dataset(promptfile=promptfile if promptfile else "", lang=language, limit=calls)
        print(f"code file saved to {csvpath}")
        
        if csvpath is not None: 
            df = pd.read_csv(csvpath, usecols=["response"])
            codebase = df["response"].dropna().astype(str).tolist()
            
            datacsv = self.code_parser.analyze(codebase, language, True)
            
            print(f"annotated file saved to {datacsv}")
            
            return str(datacsv)
            
        else:
            print("failed to generate content")
            return None

    def make_large_codebase(self, languages=["py", "cpp", "c", "java"], prompt_files=["pyprompts.txt", "cppprompts.txt", "cprompts.txt", "javaprompts.txt"], num_calls=15):
        filenames = []
        for language, prompts in zip(languages, prompt_files):
            file = self.generate_codebase(language, promptfile=prompts, calls=num_calls)
            filenames.append(file)

        print(f"annotated code files saved to {filenames}")

        return filenames

    def make_large_corpus(self, topicfile=None, bot_a=None, bot_b=None, num_topics=15, calls_per_topic=5):
        self.prose_generator.set_sysroles(bot_a, bot_b)
        topics = []

        if topicfile:
            with open(topicfile, 'r') as f:
                for line in f:
                    topics.append(line)
        else:
            gen_topics = self.topicfile
            random.shuffle(gen_topics)
            topics = gen_topics[:num_topics]
        
        filenames = []
        for topic in topics: 
            csvdata = self.generate_corpus(topic, calls_per_topic)
            if csvdata: 
                filenames.append(csvdata)

        if len(filenames) > 0:
            print(f"annotated chats saved to {filenames}")
            return filenames
        else:
            print("Chats not annotated...something went wrong.")
            return None

class dataset_builder:

    def __init__(self):
        self.languages = ["py", "cpp", "c", "java"]
        self.hf_langs = ["python", "c++", "c", "java"]
        self.code_datasets = {}
        for lang in self.hf_langs:
            self.code_datasets[lang] = load_dataset(
                    "bigcode/the-stack",
                    data_dir=f"data/{lang}",
                    split="train",
                    streaming=True
                    )
        self.brown = nltk.download("brown")
        self.ptb = nltk.download("treebank")
        self.webstream = load_dataset("openwebtext", split="train", streaming=True)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.annotator = Encorporator()
        self.code_annotator = Codecorpus()

    def build_corpus(self, categories=list(brown.categories()), num_docs=50):
        dataset = []
        
        brown_fileids = list(brown.fileids(categories=categories))
        random.shuffle(brown_fileids)

        brown_docs = []
        for fileid in brown_fileids:

            if len(brown_docs) >= num_docs:
                break
            
            brown_words = list(brown.words(fileids=fileid))
            doc_text = str(" ".join(brown_words))
            brown_docs.append(doc_text)
    
        
        dataset.extend(brown_docs)
        
        penn_fileids = list(treebank.fileids())
        random.shuffle(penn_fileids)

        penn_docs = []
        for file in penn_fileids:

            if len(penn_docs) >= num_docs:
                break
            

            file_words = [str(w) for w in treebank.words(file)]
            penn_doc = " ".join(file_words)
            penn_docs.append(penn_doc)

        dataset.extend(penn_docs)
        
        web_docs = []
        web_iter = iter(self.webstream.skip(random.randint(0, 500)))
        while len(web_docs) < num_docs: 
            try:
                sample = next(web_iter)
                raw = sample['text']
                if raw.strip():
                    web_docs.append(raw)
            except StopIteration:
                break
        
        dataset.extend(web_docs)

        datacsv = self.annotator.encorporate(dataset, llm=False)

        return datacsv

    def build_dataset(self, languages=None, num_rows=50):

        dataset = {}
        filenames = []
        codelangs = languages if languages else self.languages
        for lang, hflang in zip(codelangs, self.hf_langs):
            self.code_datasets[hflang].shuffle(buffer_size=500, seed=None)
            dataset[lang] = self.code_datasets[hflang].take(num_rows)

            codebase = [row['content'] for row in dataset[lang]]
            
            datacsv = self.code_annotator.analyze(codebase, lang, llm=False)

            if datacsv: 
                filenames.append(datacsv)

        print(f"Filenames saved: {filenames}")
        return filenames

def count_rows(csvfile):

    csv.field_size_limit(sys.maxsize)

    with open(csvfile, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        return sum(1 for row in reader)

class data_generator:
    def __init__(self, api):

        self.ai_generator = dataset_builder_ai(api_key=api)
        self.generator = dataset_builder()
        self.file_annotator = Encorporator()
        self.code_annotator = Codecorpus()
        self.magika = Magika()
        self.extensions = {
                "python": "py", "py": "py", "ipynb": "py", 
                "c++": "cpp", "cpp": "cpp", "c-plus-plus": "cpp", "c": "c", 
                "java": "java", "javascript": "js", "js": "js", "html": "html",
                "css": "css", "rust": "rust"
                }

    
    def cat_csvs(self, filenames, name=None):

        csv.field_size_limit(sys.maxsize)

        if isinstance(filenames, str): 
            path = Path(filenames)

            num_rows = count_rows(path)
            return filenames, num_rows

        if len(filenames) <= 1:
            num_rows = count_rows(filenames[0])
            return filenames[0], num_rows

        first = filenames[0]
        remaining = filenames[1:]

        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        combined = f"{name}_{timestamp}.csv"
        
        num_docs = 0
        with open(combined, 'w', encoding='utf-8') as final:
            writer = csv.writer(final)
            with open(first, 'r', encoding='utf-8') as f1:
                reader = csv.reader(f1)
                for row in reader:
                    writer.writerow(row)
                    num_docs += 1

            for file in remaining: 
                with open(file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        writer.writerow(row)
                        num_docs += 1

        return combined, num_docs-1
    
    def detect_lang(self, text):
        
        prediction = self.magika.identify_bytes(text.encode('utf-8'))
        language = prediction.output.label
        extension = self.extensions[language]

        return extension
    def generate_prose(self, name="data", sys_a=None, sys_b=None, topicsfile=None, no_topics=15, no_calls=4):

        ai_files = self.ai_generator.make_large_corpus(topicsfile, sys_a, sys_b, no_topics, no_calls)
        
        ai_dataframe, num_docs = self.cat_csvs(ai_files, f"{name}_ai_prose")

        num_docs = num_docs // 3 if num_docs > 60 else num_docs

        human_files = self.generator.build_corpus(num_docs=num_docs)

        human_dataframe, doc_count = self.cat_csvs(human_files, f"{name}_human_prose")

        print(f"{num_docs} ai prose documents saved to {ai_dataframe}")
        print(f"{doc_count} human prose documents saved to {human_dataframe}")

        return ai_dataframe, human_dataframe
    
    def generate_code(self, name="data", promptfiles=None, rows=25):

        ai_files = self.ai_generator.make_large_codebase(num_calls=rows)

        ai_dataframe, num_rows = self.cat_csvs(ai_files, f"{name}_ai_code")

        human_files = self.generator.build_dataset(num_rows=rows)

        human_dataframe, row_count = self.cat_csvs(human_files, f"{name}_human_code")

        print(f"{num_rows} ai code instances saved to {ai_dataframe}")
        print(f"{row_count} human code instances saved to {human_dataframe}")

        return ai_dataframe, human_dataframe
    
    def annotate_prose_file(self, filename, ai=False):
        if filename.endswith(".txt"):

            content = ""
            with open(filename, 'r', encoding='utf-8') as f:
                content += f.read()
                
            annotated_file = self.file_annotator.encorporate([content], ai)
            
            print(f"annotated file saved to {annotated_file}")

            return annotated_file
        else:
            print("filetype not supported")
            return None

    def annotate_code_file(self, filename, ai=False): 

        language = ""
        text = open(filename, 'r').read()

        if filename.endswith(".txt"): 
            text = open(filename, 'r').read()
            lang = self.detect_lang(text=text)
            language = self.extensions[lang]
        else:
            language = filename.split('.')[1]

        valid_langs = ["py", "ipynb", "cpp", "c", "java"]

        if language not in valid_langs:
            print(f"{language} not supported at this time")
            return None

        annotated_code = self.code_annotator.analyze_file(text, language, ai)

        print(f"annotated code saved to {annotated_code}")

        return annotated_code
    
    def augment_prose(self, filename, num_docs):
        #TODO: make function that adds the specified number of rows to the dataframe and resaves the file to the same filename





        

        


        

        

#if __name__ == "__main__":

    #ai_generator = dataset_builder_ai()
    #generator = dataset_builder()

    #ai_prosefiles = ai_generator.make_large_corpus("topics.txt", "scholar", "skeptic", 4)
    #ai_codefiles = ai_generator.make_large_codebase()
    
    #prosefiles = generator.build_corpus()
    #codefiles = generator.build_dataset()












            
    






