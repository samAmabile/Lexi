import asyncio 
import json 
import os 
from google import genai
from google.genai import types 
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CODE_DIR = DATA_DIR / "code"
LLM_DIR = CODE_DIR / "llm_code"
HUM_DIR = CODE_DIR / "human_code"
PROMPT_DIR = CODE_DIR / "promptfiles"

AGENT_SYSPROMPT = """
You are Gemini's advanced autonomous coding agent. 
Your core objective is to write production-ready, highly optimized, and clean code based on the user's prompt.

Strict Rules:
1. Return ONLY the executable code/script. Do not include markdown formatting code blocks (like ```python) unless explicitly requested.
2. No pleasantries, no conversational filler, and no post-code explanations. 
3. Include clear inline comments within the code where necessary.
4. Ensure adherence to modern best practices, proper error handling, and language-specific conventions.
"""

PROMPT_SYSPROMPT = """
You are a prompt generating bot. 
Your only objective is to write succinct and detailed prompts based on code samples submitted to you. 
You must analyze the provided code material and write a clear, detailed prompt that instructs a separate model to write a complete, standalone program achieving the same overall objective, domain, and functional intent.

Strict Rules:
1. Return ONLY the final user prompt in basic raw text format. Do not write any code, code comments, or include any snippets of the original source material. 
2. Write only plain text. Never use markdown formatting code blocks (such as ``` or ```cpp).
3. No pleasantries, no conversational filler, and no follow-up questions. Begin the response immediately with the generated prompt text.
4. If the code sample is truncated or contains '...' ellipsis segments, ignore the broken syntax. Infer the holistic intent from the available context and write a general prompt for a complete program based on that inferred domain.
5. The prompt you generate must instruct the target model to build a full, flawless script. Do not mention truncation, missing sections, or ellipses in your output.
"""


class gemini_async:
    def __init__(self, max_async = 20):
        self.model = "gemini-2.5-flash"
        self.sysprompt = AGENT_SYSPROMPT
        self.config = types.GenerateContentConfig(
                system_instruction=self.sysprompt,
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=2048
        )
        self.assistantprompt = PROMPT_SYSPROMPT
        self.prompt_config = types.GenerateContentConfig(
                system_instruction=self.assistantprompt,
                temperature=0.6, 
                top_p=0.95
        )
        self.max_concurrent = max_async
        self.client = genai.Client()
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.output_file = LLM_DIR / "gemini_code.jsonl"
        self.prompt_file = PROMPT_DIR / "gemini_prompts.jsonl"


    def limit_tokens(self, code, num_chars):

        code = str(code) if code is not None else ""

        if len(code) < num_chars:
            return code

        sections = num_chars // 3 
        first = code[:sections]
        second = code[(len(code) // 2) - (sections // 2):(len(code) // 2) + (sections // 2)]
        third = code[-sections:]

        combined = f"{first}...{second}...{third}"

        return combined

   
    async def generate_prompt(self, code, lang, lineID):

        async with self.semaphore:
            for attempt in range(3): 
                try:
                    formatted_prompt = f"Follow these instructions to produce a prompt for a model to write code in {lang}: \n{code} \n\n PRODUCE A PROMPT ONLY"
                    response = await self.client.aio.models.generate_content(
                            model=self.model,
                            contents=formatted_prompt,
                            config=self.prompt_config
                    )

                    prompt_out = response.text.strip() if response.text else ""
                    record = {"index": lineID, "source code": code, "lang": lang, "prompt": prompt_out}

                    with open(self.prompt_file, "a", encoding='utf-8') as f:
                        f.write(json.dumps(record) + '\n')

                    return True
                except Exception as e:
                    if attempt == 2:
                        record = {"index": lineID, "source code": code, "lang": "error", "prompt": f"ERROR: {str(e)}"}
                        with open(self.prompt_file, "a", encoding='utf-8') as f:
                            f.write(json.dumps(record) + '\n')
                            break
                    
                    await asyncio.sleep(3 ** (attempt+1) + random.uniform(0.5, 1.5))
            
            return False

    async def generate_code(self, prompt, lang, lineID):

        if not prompt or str(prompt).strip() == "":
            record = {"index": lineID, "prompt": prompt, "lang": 'error', "code": "ERROR: no prompt text"}
            with open(self.output_file, "a", encoding='utf-8') as f: 
                f.write(json.dumps(record) + '\n')
            
            return False
        
        async with self.semaphore:
            for attempt in range(3):
                try:
                    response = await self.client.aio.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=self.config
                    )

                    code_out = response.text.strip() if response.text else ""

                    record = {"index": lineID, "prompt": prompt, "lang": lang, "code": code_out}

                    with open(self.output_file, "a", encoding='utf-8') as f:
                        f.write(json.dumps(record) + '\n')
                    
                    return True
                except Exception as e:
                    if attempt == 2:
                        record = {"index": lineID, "prompt": prompt, "lang": "error", "code": f"ERROR: {str(e)}"}
                        with open(self.output_file, "a", encoding='utf-8') as f:
                            f.write(json.dumps(record) + '\n')
                            break
                    
                    await asyncio.sleep((5 ** (attempt+1)) + random.uniform(0.5, 1.5))

            return False
    
    async def prompt_main(self, infile):
        
        if not os.path.exists(infile):
            print(f"could not access file: {infile}")
            return

        df = pd.read_csv(infile)
        df['src'] = df['src'].fillna("").astype(str)
        clean_df = df[~(df['src'].astype(str).str.startswith('NORECORD') | (df['lang'].astype(str) == 'error'))].copy()
        clean_df['src'] = clean_df['src'].apply(lambda x: self.limit_tokens(x, num_chars=3000))

        sourcecode = list(zip(clean_df.index, clean_df['lang'], clean_df['src']))

        tasks = [
            self.generate_prompt(code=code, lang=lang, lineID=int(i))
            for i, lang, code in sourcecode
        ]

        await tqdm_asyncio.gather(*tasks, desc="Generating prompts from source code")

        print(f"\nSuccess: all generated prompts saved to {self.prompt_file}")

    async def main(self, infile): 

        if not os.path.exists(infile): 
            print(f"Could not access file : {infile}")
            return 
        
        

        succesful_indices: set[int] = set()

        if os.path.exists(self.output_file):
            try:
                out_df = pd.read_json(self.output_file, lines=True)
                if not out_df.empty and "index" in out_df.columns:
                    success_df = out_df[out_df["lang"] != "error"]
                    succesful_indices = set(success_df["index"].astype(int).tolist())
            except Exception as e:
                print(f"Either no succesful rows exist or there was an issue accessing the path: {str(e)}")


        df = pd.read_json(infile, lines=True)
        valid_df = df.dropna(subset=["prompt"])

        clean_df = valid_df[~valid_df["index"].astype(int).isin(list(succesful_indices))]

        if clean_df.empty:
            print("All prompts already succesfully processed")
            return 

        print(f"found {len(clean_df)}/{len(valid_df)} prompts still needing processing")

        prompts = list(zip(clean_df["prompt"], clean_df["lang"], clean_df["index"]))


        tasks = [
            self.generate_code(prompt, lang, index) 
            for prompt, lang, index in prompts
        ]

        await tqdm_asyncio.gather(*tasks, desc="Generating code from promptfile")

        print(f"\nSuccess: all generated code saved to {self.output_file}")

        self.deduplicate_rows(self.output_file)
    
    def prompt_run(self, infile): 

        asyncio.run(self.prompt_main(infile))

    def run(self, infile):

        asyncio.run(self.main(infile))

    def deduplicate_rows(self, filepath):
        if os.path.exists(filepath):
            df = pd.read_json(filepath, lines=True)

            df["failed"] = df["lang"] == "error"
            df = df.sort_values('failed', ascending=False)
            df = df.drop_duplicates(subset=["index"], keep="last").drop(columns = ["failed"])
            
            df.to_json(filepath, orient='records', lines=True)
            print(f"Cleaned up duplicates and saved corrected data to {filepath}")


if __name__ == "__main__":
    


    generator = gemini_async(max_async=8)
    #generator.prompt_run(CODE_DIR / "new_webscraped_code.csv")
    generator.run(PROMPT_DIR / "gemini_prompts.jsonl")












