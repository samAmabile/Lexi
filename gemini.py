#GEMINI API ACCESS FOR LLM TEXT GENERATION (PROSE AND CODE) 
from google import genai
from google.genai import types
import pandas as pd

import csv
import os 
from datetime import datetime
import random 
import time
from pathlib import Path
from magika import Magika
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

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

stop_words = set(stopwords.words('english'))

CHATLOGS = "chatlogs"
BOTLOGS = "botlogs"
CODELOGS = "codelogs"

class Chat:

    def __init__(self, API, model="gemini-2.5-flash", sys="you are succinct and helpful assistant"):
        self.api = API
        self.sys = sys

        self.model = model
        self.client = self.gemini_connect()
        self.config = types.GenerateContentConfig(
            system_instruction = self.sys
        )
        self.history : list[types.Content] = []

        self.create_master()
    def gemini_connect(self): 
        try:
            gemini_client = genai.Client(api_key=self.api)
            return gemini_client
        except Exception as e:
            print(f"Error, unable to connect to Gemini: {e}")
            return None

    def verify_path(self, folder):
        os.makedirs(folder, exist_ok=True)

    def create_master(self):
        self.verify_path(CHATLOGS)
        master = Path(CHATLOGS) / "master.csv"
        if not master.exists():
            with open(master, 'w', encoding='utf-8', newline='') as m:
                writer = csv.writer(m)
                writer.writerow(['Role', 'Content'])
        return 

    def save_chat(self, chatname):
        filename = chatname+".csv" if not chatname.endswith(".csv") else chatname
        self.verify_path(CHATLOGS)
        path = Path(CHATLOGS) / filename
        master = Path(CHATLOGS) / "master.csv"
        with open(path, 'w', encoding='utf-8', newline='') as f, open(master, 'a', encoding='utf-8', newline='') as m: 
            writer = csv.writer(f)
            master_writer = csv.writer(m)

            writer.writerow(['Role', 'Content'])
            for message in self.history:
                if message and message.role and message.parts:
                    role = message.role.capitalize()
                    content = message.parts[0].text if message.parts[0].text else ""
                    writer.writerow([role, content])
                    master_writer.writerow([role, content])

        return path

    def chat_loop(self):
        if not self.client:
            print("Not connected to API")
            return
        
        try:
            chat = self.client.chats.create(
                model=self.model,
                config=self.config
            )
        except Exception as e:
            print(f"Error, could not initiate chat: {e}")
            return
        
        print(f"\nConnected to {self.model}")
        print("\nType 'Exit' to terminate chat and save to file")
        
        hasTitle = False
        title = "untitled_chat"

        while True:
            prompt = input("\nYou: ").strip()
            if prompt in ["Exit", "exit"]:
                break
            
            
            if not hasTitle:
                title_words = prompt.split()
                if len(title_words) > 3: 
                    title = "_".join([w for w in title_words if w not in stop_words])
                else:
                    title = "_".join(title_words)
                
                hasTitle = True

            try:
                response = chat.send_message(prompt)
                gemini_reply = response.text
                print(f"\nGemini: {gemini_reply}")
            except Exception as e:
                print(f"API connection timed out: {e}")
                break
        
        self.history = chat.get_history()

        if self.history:
            csvname = self.save_chat(title+".csv")
            return csvname

        return None

class Automate:

    def __init__(self, API, sys_a="you are a succinct and helpful assistant", sys_b="you are an inquisitive mind", model_a="gemini-3.1-flash-lite", model_b="gemini-3.1-flash-lite"):
        self.api = API

        self.sys_a = "you are a(n) "+sys_a if not sys_a.startswith("you are a") else sys_a
        self.sys_b = "you are a(n) "+sys_b if not sys_b.startswith("you are a") else sys_b

        self.model_a = model_a
        self.model_b = model_b

        self.client = self.gemini_connect()
        self.history: list[types.Content] = []

        self.topics = [line for line in open("bottopics.txt", 'r')]

        self.create_master()
    
    def set_sysroles(self, role_a=None, role_b=None):
        if role_a and role_b:
            self.sys_a = f"you are a(n) {role_a}" if not role_a.startswith("you are a") else role_a
            self.sys_b = f"you are a(n) {role_b}" if not role_b.startswith("you are a") else role_b
        elif role_a:
            self.sys_a = f"you are a(n) {role_a}" if not role_a.startswith("you are a") else role_a
            self.sys_b = f"you are a(n) {role_a}" if not role_a.startswith("you are a") else role_a
        elif role_b:
            self.sys_a = f"you are a(n) {role_b}" if not role_b.startswith("you are a") else role_b
            self.sys_b = f"you are a(n) {role_a}" if not role_b.startswith("you are a") else role_b

    def gemini_connect(self): 
        try:
            gemini_client = genai.Client(api_key=self.api)
            return gemini_client
        except Exception as e:
            print(f"Error, unable to connect to Gemini: {e}")
            return None

    def verify_path(self, folder):
        os.makedirs(folder, exist_ok=True)

    def create_master(self):
        self.verify_path(BOTLOGS)
        master = Path(BOTLOGS) / "master.csv"
        if not master.exists():
            with open(master, 'w', encoding='utf-8', newline='') as m:
                writer = csv.writer(m)
                writer.writerow(['Role', 'Content'])
        return 

    def save_chat(self, chatname):
        filename = chatname+".csv" if not chatname.endswith(".csv") else chatname
        self.verify_path(BOTLOGS)
        path = Path(BOTLOGS) / filename
        master = Path(BOTLOGS) / "master.csv"
        with open(path, 'w', encoding='utf-8', newline='') as f, open(master, 'a', encoding='utf-8', newline='') as m: 
            writer = csv.writer(f)
            master_writer = csv.writer(m)

            writer.writerow(['Role', 'Content'])
            for message in self.history:
                if message and message.role and message.parts:
                    role = message.role
                    content = message.parts[0].text
                    writer.writerow([role, content])
                    master_writer.writerow([role, content])

        return path

    def automate_loop(self, topic=None, iterations=6):
        if not self.client:
            print("not connected to API, bot chat cancelled")
            return

        if not topic: 
            random.shuffle(self.topics)
            topic = self.topics[0]

        chat_a = self.client.chats.create(
                model = self.model_a,
                config = types.GenerateContentConfig(system_instruction=self.sys_a, response_mime_type="text/plain")
        )
        chat_b = self.client.chats.create(
                model = self.model_b,
                config = types.GenerateContentConfig(system_instruction=self.sys_b, response_mime_type="text/plain")
        )

        print(f"-----------Starting chat on {topic}--------------")

        prompt = f"Lets discuss {topic}. What are your thoughts?"
        
        try:
            for i in range(iterations):
                
                response_a = chat_a.send_message(prompt)
                text_a = response_a.text
                print(f"Bot_A: {text_a}")

                response_b = chat_b.send_message(text_a)
                text_b = response_b.text
                print(f"Bot_B: {text_b}")

                prompt = text_b

                time.sleep(1)

        except Exception as e: 
            print(f"connection lost: {e}")


        self.history = chat_a.get_history()
        print("----------------chat complete----------------------")

        if self.history is not None:
            title = topic.replace(" ", "_")
            csvname = self.save_chat(f"{title}_bot_chat.csv")
            return str(csvname)

        return None

class Code:

    def __init__(self, API, model="gemini-3.1-flash-lite"):
        self.client = genai.Client(api_key=API)
        self.model = model

        self.magika = Magika()

        with open("engine_rules.txt", 'r') as f: 
            sys_instruct = f.read()

        self.config = types.GenerateContentConfig(
                system_instruction=sys_instruct,
                temperature=0.2,
                response_mime_type="text/plain"
        )

        self.verify_path(CODELOGS)

    def verify_path(self, folder):
        os.makedirs(folder, exist_ok=True)

    def detect_lang(self, text):
        #lexer = guess_lexer(text)

        prediction = self.magika.identify_bytes(text.encode('utf-8'))

        language = prediction.output.label
        
        #language = lexer.name
        extension = self.get_extension(language)

        return extension

    def get_extension(self, language):
        lang = language.lower().strip()
        keys = {
            "python": "py", "py": "py", "python3": "py",
            "cpp": "cpp", "c++": "cpp", "cc": "cpp",
            "javascript": "js", "js": "js", "node": "js",
            "typescript": "ts", "ts": "ts",
            "html": "html", "htm": "html",
            "css": "css", "bash": "sh", "shell": "sh", 
            "java": "java", "gdscript": "cpp", "gd": "cpp",
            "c": "c", "rust": "rs", "rs": "rs", "ruby": "rb", 
            "rb": "rb", "r": "r"
        }

        return keys.get(lang, lang)

    def generate_title(self, prompt): 
        if not prompt: 
            return datetime.now().strftime('%Y%m%d_%H%M%S')
        
        clean = re.sub(r'[^\w\s]', '', prompt.lower())
        words = [w for w in clean.split() if w not in stop_words and len(w) > 2]

        if not words: 
            return datetime.now().strftime('%Y%m%d_$H$M$S')
        
        fdist = nltk.FreqDist(words)
        top_n = [w for w, count in fdist.most_common(5)]

        title = "_".join(top_n)[:5]

        return title

    def generate_lemma_title(self, text):

        if not text:
            return datetime.now().strftime('%Y-%m-%d_%H%M')

        if len(text.split()) < 6:
            words = text.split()
            text = " ".join([w for w in words if w.lower() not in stop_words])
            text = re.sub(r'[^\w\s-]', '', text.lower()).replace(' ', '_')
            if not text: 
                return datetime.now().strftime('%Y-%m-%d_%H%M')
            return text

        words = re.findall(r'\w+', text.lower())

        tagged = nltk.pos_tag([w for w in words if w not in stop_words])
        lemmas = [lemmatizer.lemmatize(w, get_pos(t)) for w, t in tagged]

        fdist = nltk.FreqDist(lemmas)

        title_set = set([w for w, count in fdist.most_common(5)])

        title = "_".join([w for w in words if w in title_set])

        return title

    def generate_filename(self, response_text):

        filenaming_config = types.GenerateContentConfig(
                system_instruction="You are a file-naming utility bot. Only output single word filenames for provided code. If multiple words are required NEVER use ' ' (blank space) ALWAYS use '_' (underscore). (e.g. script.js, filename.py, data_utility.cpp, binary_search.c)",
                temperature=0.0,
                response_mime_type="text/plain"
        )

        prompt =f"Generate a single word filename for this code. Use '_' for spaces and include the appropriate extension (.py, .hpp, .c, .cpp, .html, .js, .ts, .sh) for the coding language. ONLY PRODUCE A ONE WORD FILENAME: {response_text}"

        response = self.client.models.generate_content(
                model=self.model,
                contents=prompt, 
                config=filenaming_config
        )

        return response.text

    def generate_promptfile(self, language, number=20): 
        promptlist_generator = types.GenerateContentConfig(
                system_instruction=f"You are a prompt generating utility bot. Generate a list of {number} prompts for prompting an LLM to generate code in the provided coding language: {language}. \nDO NOT GENERATE CODE \nDO NOT GENERATE ANY OTHER CONTENT \nONLY GENERATE A LIST OF PROMPTS FOR THE SPECIFIED LANGUAGE",
                temperature=0.7, 
                response_mime_type="text/plain"
        )

        prompt = f"Please provide a list of {number} varied prompts for prompting an LLM to produce code in {language}. \nPrompts should be specific and concise. \nOne prompt per line \nFor example : 'Create a crossword puzzle generator in {language}', 'Write code to parse names and dates from a textfile in {language}', 'Build a whiteboard application for desktop in {language}'"

        response = self.client.models.generate_content(
                model=self.model, 
                contents=prompt, 
                config=promptlist_generator
        )

        return response.text

    def save_code(self, text, filename):

        self.verify_path(CODELOGS) 
        path = Path(CODELOGS) / filename
        with open(path, 'w', encoding='utf-8') as f: 
            f.write(text)

        return str(path)

    def save_history(self, history, filename): 
        
        filename = filename+".csv" if not filename.endswith(".csv") else filename
        self.verify_path(CODELOGS)


        path = Path(CODELOGS) / filename
        master = Path(CODELOGS) / "codemaster.csv"
        with open(path, 'w', encoding='utf-8', newline='') as f, open(master, 'a', encoding='utf-8', newline='') as m: 
            writer = csv.writer(f)
            master_writer = csv.writer(m)

            writer.writerow(['Role', 'Content'])
            for message in history:
                if message and message.role and message.parts:
                    role = message.role.capitalize()
                    content = message.parts[0].text
                    writer.writerow([role, content])
                    master_writer.writerow([role, content])

        return path

    def append_master(self, dataframe): 
        master_path = Path(CODELOGS) / "master_dataset.csv"

        header_check = master_path.exists()

        dataframe.to_csv(master_path, mode='a', index=False, header=not header_check)

    def generate_code(self, prompt, language="python"):
        
        formatted_prompt = f"LANGUAGE:{language}\nTASK:{prompt}\nONLY OUTPUT RAW CODE"

        response = self.client.models.generate_content(
                model=self.model, 
                contents=formatted_prompt, 
                config=self.config
        )
        
        extension = self.get_extension(language)
        filename = self.generate_filename(response.text)
        location = self.save_code(response.text, filename)

        return extension, location, response.text

    def debug(self, code, changes=""):

        parsing_config = types.GenerateContentConfig(system_instruction="you are a code correcting and parsing utility bot. your directives are:\n1. find and correct bugs in code that is given to you\n2. make changes and improvements to code as requested\n3. generate only raw code, no explanations or other prose except as comments\nDO NOT IGNORE DIRECTIVES. DO NOT GENERATE PROSE, ONLY RAW CODE",
                temperature=0.2, 
                response_mime_type="text/plain"
        )

        change_prompt = f"and make the following changes: {changes}"

        formatted_prompt = f"PRODUCING ONLY CODE debug and correct this code{change_prompt if changes != "" else ""}: {code}"

        response = self.client.models.generate_content(
                model=self.model,
                contents=formatted_prompt, 
                config=parsing_config
        )

        return response.text

    def automate_code(self, prompt, iterations=6, lang=None):

        prompting_config = types.GenerateContentConfig(
                system_instruction="you are a prompt generating utility bot for a code generation system. your directives are:\n1. generate a short prompt with instructions for writing code based on the input you are given\n2. if the input you are given is code, generate instructions for new modules to expand on the provided code\n3. DO NOT comment on or give feedback to prompts ONLY produce new prompts\n4. DO NOT produce code, only instructions for code production\n DO NOT IGNORE DIRECTIVES. ONLY GENERATE PROMPT",
                temperature = 0.3,
                response_mime_type="text/plain"
        )
        
        lang_instruction = f" in {lang}" if lang else ""
        formatted_prompt = f"Generate a short prompt with instructions to create code{lang_instruction} based on this request: {prompt}"

        #updated title-making logic from member function:
        title = self.generate_lemma_title(prompt)

        prompt_bot = self.client.chats.create(
                model=self.model,
                config=prompting_config
        )

        code_bot = self.client.chats.create(
                model=self.model, 
                config=self.config
        )
        
        code_content = ""
        session_data = []
        label = "llm_code"

        for i in range(iterations):

            new_prompt = prompt_bot.send_message(formatted_prompt)
            prompt_text = new_prompt.text
            print(f"PROMPT: {prompt_text}")

            code_response = code_bot.send_message(prompt_text)
            code_text = code_response.text or ""
            print(f"CODE: {code_text}")
            
            code_content += f"\n\n{code_text}"
            
            if lang:
                language = self.get_extension(lang)
            else:
                language = self.detect_lang(code_text)

            if language in ["gd", "gdscript"]:
                if "#include" in code_text or ";" in code_text:
                    language = "cpp"
                elif "import" in code_text or "def" in code_text:
                    language = "py"
            
            session_data.append({
                "iteration": i+1, 
                "prompt": prompt_text, 
                "response": code_text, 
                "model": self.model,
                "language": language,
                "label": label,
                "timestamp": datetime.now().isoformat()
                })

            formatted_prompt = f"Generate a short prompt for further additions to this code: \n{code_text}. \nDO NOT GENERATE CODE, ONLY GENERATE A PROMPT FOR ADDITIONAL CODE"
            time.sleep(1)
        
        df = pd.DataFrame(session_data)
        df_path = Path(CODELOGS) / f"{title}_dataset.csv"
        df.to_csv(df_path, index=False)

        extension = self.detect_lang(code_content)
        if extension in ["gd", "gdscript"]:
            if "#include" in code_content or ";" in code_content:
                language = "cpp"
            elif "import" in code_content or "def" in code_content:
                language = "py"

        code_file = self.save_code(code_content, f"{title}.{extension}")
        self.append_master(df)
        #csv = self.save_history(prompt_bot.get_history(), f"{title}.{extension}")

        return str(df_path), code_file
    
    def generate_code_dataset(self, promptfile="", lang="py", limit=None):

        all_prompts = []

        if promptfile != "":
            with open(promptfile, 'r') as f: 
                for line in f:
                    all_prompts.append(line)
        else:
            num_prompts = limit if limit else 20
            promptlist = self.generate_promptfile(lang, num_prompts)
            all_prompts = promptlist.splitlines() if promptlist else []
            if len(all_prompts) < 1:
                print("failed to generate prompt list")
                return None
            elif len(all_prompts) < 2: 
                print("prompts not separated by newline")
                return None

        random.shuffle(all_prompts)

        prompts = all_prompts[:limit] if limit else all_prompts

        dataset = []
        label = "llm_code"
        for i, prompt in enumerate(prompts): 
            try:
                response = self.client.models.generate_content(model=self.model, contents=prompt, config=self.config)

                response_text = response.text or ""

                language = lang

                dataset.append({
                    "iteration": i+1,
                    "prompt": prompt,
                    "response": response_text,
                    "model": self.model, 
                    "language": language,
                    "label": label,
                    "timestamp": datetime.now().isoformat()
                    })
                time.sleep(1)
            except Exception as e:
                print(f"Skipped prompt due to error: {e}")

        
        
        date = datetime.now()
        formatted_date = date.strftime('%Y-%m-%d_%H%M')
        title = f"code_dataset_{formatted_date}.csv"

        path = Path(CODELOGS) / title

        df = pd.DataFrame(dataset)
        df.to_csv(path, index=False, encoding='utf-8')
        self.append_master(df)

        return str(path)

#if __name__ == "__main__":

#    API = os.environ.get("GEMINI_API_KEY")
#    generator = Code(API)

#    automated_chat = generator.automate_code("a secure task management application in cpp", 4)

#    test_dataset = generator.generate_code_dataset(lang='py', limit=5)

#    print(f"Automated code production process saved to {automated_chat}, test_dataset saved to {test_dataset}")








