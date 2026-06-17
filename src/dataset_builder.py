from gemini import Automate, Code, Chat
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
from google.genai.errors import ServerError
import textwrap
import asyncio
from collections import defaultdict
from google.genai.errors import ClientError


import nltk 
from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.corpus import webtext
from nltk import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from transformers import AutoTokenizer
from magika import Magika

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LANG_DIR = DATA_DIR / "language"
CODE_DIR = DATA_DIR / "code"
LANGPROMPTS = LANG_DIR / "promptfiles"
CODEPROMPTS = LANG_DIR / "promptfiles"
LLM_LANG = LANG_DIR / "llm_language"
HUM_LANG = LANG_DIR / "natural_language"
LLM_CODE = CODE_DIR / "llm_code"
HUM_CODE = CODE_DIR / "human_code"
CODE_PROMTPS = CODE_DIR / "promptfiles"
LANG_PROMPTS = LANG_DIR / "promptfiles"


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
        self.topicfile = [line for line in open(LANG_PROMPTS / "bottopics.txt", 'r')]

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

    def make_large_codebase(
            self, 
            languages=["py", "cpp", "c", "java"], 
            prompt_files=[
                str(CODE_DIR / "promptfiles/pyprompts.txt"), 
                str(CODE_DIR / "promptfiles/cppprompts.txt"), 
                str(CODE_DIR / "promptfiles/cprompts.txt"), 
                str(CODE_DIR / "promptfiles/javaprompts.txt")], 
            num_calls=15):
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
        self.codebase = []
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

    def summarize_text(self, text): 
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, 3)
        return " ".join(str(s) for s in summary)

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

        prompts = []

        for i, document in enumerate(dataset):
            doc = document.strip()
            tokens = doc.split()
            num_tokens = len(tokens)
            if not doc or num_tokens < 10:
                print(f"Skipping document {i+1} too short or empty")
                continue

            try:
                summary = self.summarize_text(doc)
                if not summary.strip():
                    print(f"Skipping {i+1}, vector calculation failed")
                    continue

            except Exception as e:
                print(f"Error summarizing document {i+1}: {e}")
                continue

            prompt = f"A document about: {summary}"
            prompts.append((prompt, num_tokens))

        datacsv = self.annotator.encorporate(dataset, llm=False)
        
        return datacsv, prompts

    def is_valid_sample(self, script):
        """returns false if the script is exessively long or has excessive characters per line"""
        if not script or not script.strip():
            return False

        if len(script.encode('utf-8')) > 100000:
            return False

        lines = script.splitlines()
        if any(len(line) > 800 for line in lines):
            return False
        
        num_lines = len(lines)

        if num_lines > 1500:

            assigns = script.count('=')
            if assigns / num_lines < 0.05: 
                return False

            if any(keyword in script for keyword in ["include", "import"]):
                semicolons = script.count(';')
                if semicolons > 0 and (semicolons / num_lines) < 0.10: 
                    return False

        return True

    def build_async_dataset(self, num_rows=500):
        dataset = {}
        filenames = []
        full_codebase = []
        all_exts = []
        codelangs = self.languages

        for lang, hflang in zip(codelangs, self.hf_langs):

            shuffled_data = self.code_datasets[hflang].shuffle(buffer_size=500, seed=None)

            data_iter = iter(shuffled_data)
            
            codebase = []
            print(f"streaming {num_rows} scripts from HF in {lang}, then applying automation filter...")

            while len(codebase) < num_rows:
                try:
                    row = next(data_iter)
                    content = row['content']

                    if not self.is_valid_sample(content):
                        continue

                    codebase.append(content)
                except StopIteration:
                    print("Ran out of files in stream, iteration stopped.")
                    break
            
            full_codebase.extend(codebase)
            actual_rows = len(codebase)
            exts = [lang] * actual_rows
            all_exts.extend(exts)

            datacsv = self.code_annotator.analyze(codebase, lang, llm=False)
            if datacsv:
                filenames.append(datacsv)

        print(f"Filenames saved: {filenames}")
        self.codebase = full_codebase
        return filenames, all_exts

    def build_dataset(self, languages=None, num_rows=50):

        dataset = {}
        filenames = []
        full_codebase = []
        all_exts = []
        codelangs = languages if languages else self.languages
        for lang, hflang in zip(codelangs, self.hf_langs):
            self.code_datasets[hflang].shuffle(buffer_size=500, seed=None)
            dataset[lang] = self.code_datasets[hflang].take(num_rows)

            codebase = [row['content'] for row in dataset[lang]]
            full_codebase.extend(codebase)
            exts = [lang] * num_rows
            all_exts.extend(exts)
            
            datacsv = self.code_annotator.analyze(codebase, lang, llm=False)

            if datacsv: 
                filenames.append(datacsv)

        print(f"Filenames saved: {filenames}")
        self.codebase = full_codebase
        return filenames, all_exts

def count_rows(csvfile):

    csv.field_size_limit(sys.maxsize)

    with open(csvfile, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        return sum(1 for row in reader)

class data_generator:
    def __init__(self, api):

        self.ai_generator = dataset_builder_ai(api_key=api)
        self.ai_prose_generator = Chat(API=api)
        self.ai_code_generator = Code(api)
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
        self.codebase = self.generator.codebase

    async def process_batch(self, data_generator, prompts, max_simultaneous=10):
        """
        process batches of prompts to LLM simultaneously
        Args:
            data_generator: the LLM client instance
            prompts: the list of (prompt, language) tuples 
            max_simultaneous: the limit of concurrent calls allowed (default 10)
        """
        semaphore = asyncio.Semaphore(max_simultaneous)
        
        global_429_limit = 50
        global_429_count = 0
        abort = asyncio.Event()
        async def caller(prompt, lang):
            
            nonlocal global_429_count
            
            async with semaphore:
                max_attempts = 5
                delay = 3
                for attempt in range(max_attempts):

                    if abort.is_set():
                        return {"prompt": prompt, "lang": lang, "result": "Aborted due to budget safety limit.", "status": "failure"}
                    try:
                        await asyncio.sleep(0.1)
                        result = await data_generator.generate_async_code(prompt, lang)
                        return {"prompt": prompt, "lang": lang, "result": result, "status": "success"}
                    except ClientError as e: 
                        error = str(e)
                        is_429 = getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in error
                        if is_429 and attempt < max_attempts-1:
                            global_429_count += 1

                            if global_429_count >= global_429_limit:
                                print("\nReached global attempt limit, aborting")
                                abort.set()
                            sleep = delay + random.uniform(0, 5)
                            print(f"\n[429 Quota Reached] Cooling down caller")
                            print(f" ->attempt {attempt}/{max_attempts}")

                            await asyncio.sleep(sleep**attempt)
                            continue
                        return {"prompt": prompt, "lang": lang, "result": str(e), "status": "failure"}
                    except Exception as e:
                        return {"prompt": prompt, "lang": lang, "result": str(e), "status": "failure"}
                return {"prompt": prompt, "lang": lang, "result": "Max retries exceeded under 429 constraints.", "status": "failure"}
        
        tasks = [caller(prompt, lang) for prompt, lang in prompts]

        result = await asyncio.gather(*tasks)
        return result

    
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

        human_files, prompts = self.generator.build_corpus(num_docs=num_docs)

        human_dataframe, doc_count = self.cat_csvs(human_files, f"{name}_human_prose")

        print(f"{num_docs} ai prose documents saved to {ai_dataframe}")
        print(f"{doc_count} human prose documents saved to {human_dataframe}")

        return ai_dataframe, human_dataframe
    
    def generate_topic_match_prose(self, num_docs=30):

        human_dataset, prompts = self.generator.build_corpus(num_docs=num_docs)
        ai_responses = []
        for prompt, num_tokens in prompts:
            if prompt != "":
                try:
                    response = self.ai_prose_generator.generate_prose(prompt, num_tokens)
                    ai_responses.append(response)
                except ServerError as e:
                    print(f"\nServer not available, skipping this call. Error: {e}")
                    time.sleep(5)
                    continue

            else:
                continue

        crashproof_ai_responses = [r for r in ai_responses if r and str(r).strip()]

        ai_dataset = self.file_annotator.encorporate(crashproof_ai_responses, True)

        return ai_dataset, human_dataset

    def generate_topic_match_code(self, name, num_scripts=30):
        human_files, exts = self.generator.build_dataset(num_rows=num_scripts)

        prompt = textwrap.dedent("""\
            Rewrite the code sample below or some approximation of it in your own style. 
            DO NOT copy the code verbatim.
            rewrite the logic from scratch with whatever changes and improvements you see fit. 
            Do not copy the method/function names, but base your function names off them. 
            CRITICAL RULES:
            1. Do not copy lines of code verbatim. Change the structural patterns, loops, and logic flow where appropriate.
            2. You may keep the core functionality and variables so the code remains functionally equivalent.
            5. Do not change languages. Produce code in the language of the sample.
            6. Maintain the same scope and feature completeness as the sample. Do not truncate the functionality or omit major logic structures.

            Code Sample Below
            """)

        prompt_2 = "Rewrite the code in your own style"
        prompt_3 = "Improve on this code by rewriting it. Dont keep anything as is, rewrite the whole thing"
        prompt_4 = "Write a program similar to this one."

        prompts = [prompt, prompt_2, prompt_3, prompt_4]
        
        chosen_prompt = random.choice(prompts)
        self.codebase = self.generator.codebase

        py_scripts = []
        c_scripts = []
        java_scripts = []
        cpp_scripts = []

        for script, lang in zip(self.codebase, exts):
            #lang = self.detect_lang(script)
            formatted_prompt = f"{chosen_prompt}: \n{script}"
            code = self.ai_code_generator.generate_code(formatted_prompt, lang)
            if code:
                if lang == 'py':
                    py_scripts.append(code)
                elif lang == 'c':
                    c_scripts.append(code)
                elif lang == 'cpp':
                    cpp_scripts.append(code)
                elif lang == 'java':
                    java_scripts.append(code)
        
        py_file = self.code_annotator.analyze(py_scripts, 'py', True)
        c_file = self.code_annotator.analyze(c_scripts, 'c', True)
        cpp_file = self.code_annotator.analyze(cpp_scripts, 'cpp', True)
        java_file = self.code_annotator.analyze(java_scripts, 'java', True)

        ai_files = [py_file, c_file, cpp_file, java_file]

        human_csv, num_rows = self.cat_csvs(human_files, f"{name}_human_code")
        ai_csv, row_count = self.cat_csvs(ai_files, f"{name}_ai_code")

        
        return human_csv, num_rows, ai_csv, row_count
    
    def is_valid_sample(self, script):
        """returns false if the script is exessively long or has excessive characters per line"""
        if not script or not script.strip():
            return False

        if not isinstance(script, str):
            return False
        
        if len(script.encode('utf-8')) > 100000:
            return False
        
        if isinstance(script, str) and script.startswith('{') and script.endswith('}') and "RESOURCE_EXHAUSTED" in script:
            print("Resource exhausted, API calls will fail")
            return False

        lines = script.splitlines()
        if any(len(line) > 800 for line in lines):
            return False
        
        num_lines = len(lines)

        if num_lines > 1500:

            assigns = script.count('=')
            if assigns / num_lines < 0.05: 
                return False

            if any(keyword in script for keyword in ["include", "import"]):
                semicolons = script.count(';')
                if semicolons > 0 and (semicolons / num_lines) < 0.10: 
                    return False

        return True

    def generate_async_topic_match_code(self, name, num_scripts=500):
        human_files, exts = self.generator.build_async_dataset(num_rows=num_scripts)
        self.codebase = self.generator.codebase
        prompt_list = list(zip(self.codebase, exts))

        max_callers = 15

        print(f"Initiating {num_scripts} async calls to LLM")

        all_responses = asyncio.run(
                self.process_batch(self.ai_code_generator, prompt_list, max_simultaneous=max_callers)
        )
        
        generated_scripts = []
        for resp in all_responses:
            if resp["status"] == "success":
                lang = resp["lang"]
                script = resp["result"]
                if not script or not isinstance(script, str) or not script.strip():
                    print(f"Warning: Empty or invalid script returned for {lang}. Skipping.")
                    continue
                if not self.is_valid_sample(script=script):
                    print(f"caught invalid code produced by LLM. looks like: {script[:100]}")
                    continue
                if script and len(script.split()) > 10:
                    generated_scripts.append((script, lang))
            else:
                print(f"error generating {resp['lang']}: {resp['result']}")
        
        scripts_by_lang = defaultdict(list)
        for script, lang in generated_scripts:
            scripts_by_lang[lang].append(script)

        py_file = self.code_annotator.analyze(scripts_by_lang["py"], 'py', True)
        c_file = self.code_annotator.analyze(scripts_by_lang["c"], 'c', True)
        cpp_file = self.code_annotator.analyze(scripts_by_lang["cpp"], 'cpp', True)
        java_file = self.code_annotator.analyze(scripts_by_lang["java"], 'java', True)

        ai_files = [py_file, c_file, cpp_file, java_file]

        human_csv, num_rows = self.cat_csvs(human_files, f"{name}_human_code")
        ai_csv, row_count = self.cat_csvs(ai_files, f"{name}_ai_code")

        return human_csv, num_rows, ai_csv, row_count


    def generate_code(self, name="data", promptfiles=None, rows=25):

        ai_files = self.ai_generator.make_large_codebase(num_calls=rows)

        ai_dataframe, num_rows = self.cat_csvs(ai_files, f"{name}_ai_code")

        human_files, exts = self.generator.build_dataset(num_rows=rows)

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
    
    def augment_ai_prose(self, filename, num_docs):
        num_calls = num_docs // 2 

        filenames = self.ai_generator.make_large_corpus(num_topics=num_calls, calls_per_topic=1)
        df = pd.read_csv(filename)
        new_file = filename.split('.')[0]
        new_data, num_docs = self.cat_csvs(filenames, f"{new_file}_copy")

        combined, total_docs = self.cat_csvs([filename, new_data], new_file)

        return combined, total_docs

    def augment_human_prose(self, filename, num_docs):

        num_calls = num_docs // 3

        new_file = self.generator.build_corpus(num_docs=num_calls)

        new_filename = filename.split('.')[0]

        combined, total_docs = self.cat_csvs([filename, new_file], new_filename)

        return combined, total_docs











        

        


        

        

#if __name__ == "__main__":

    #ai_generator = dataset_builder_ai()
    #generator = dataset_builder()

    #ai_prosefiles = ai_generator.make_large_corpus("topics.txt", "scholar", "skeptic", 4)
    #ai_codefiles = ai_generator.make_large_codebase()
    
    #prosefiles = generator.build_corpus()
    #codefiles = generator.build_dataset()












            
    






