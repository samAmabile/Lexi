import re
import pandas as pd
import json 
import numpy 
from pathlib import Path
import os

from encorporate import Codecorpus

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CODE_DIR = DATA_DIR / "code"
LLM_DIR = DATA_DIR / "llm_code"
HUM_DIR = DATA_DIR / "human_code"
PROMPT_DIR = DATA_DIR / "promptfiles"

class annotator:
    def __init__(self): 
        self.codecorpus = Codecorpus()
        self.langs = {'py', 'cpp', 'c', 'java'}

    def load_df_column(self, df, column, astype=str): 

        return df[column].dropna().astype(str).tolist()
    
    def groupby_lang(self, codebase): 

        patterns = {
            'py': re.compile(r'^\s*(def\s+|import\s+|from\s+\w+\s+import|if\s+__name__\s*==|print\()', re.M),
            'java': re.compile(r'(public\s+class\s+|System\.out\.print|public\s+static\s+void\s+main)', re.M),
            'cpp': re.compile(r'(#include\s*<iostream>|std::|cout\s*<<|using\s+namespace\s+std)', re.M),
            'c': re.compile(r'(#include\s*<stdio\.h>|printf\(|int\s+main\s*\()', re.M)
        }

        buckets = {'py': [], 'cpp': [], 'c': [], 'java': []}

        for script in codebase: 

            matched = False

            for lang in {'py', 'cpp', 'c', 'java'}:
                if patterns[lang].search(script) is not None: 
                    buckets[lang].append(script)
                    
                    matched = True

            if not matched: 
                if ';' in script and ('{' in script or '}' in script): 

                    c_score = 0
                    cpp_score = 0

                    c_pattern = re.compile(r'\s*(printf|scanf|malloc|free|stio.h|sizeof|typedef|restrict|)')
                    cpp_pattern = re.compile(r'\s*(class|public|private|new|cout|cin|std|template|typename|::|&|iostream|cerr|endl|)')

                    c_matches = c_pattern.findall(script)
                    cpp_matches = cpp_pattern.findall(script)

                    c_score += len(c_matches)
                    cpp_matches = len(cpp_matches)

                    if cpp_score > c_score:
                        buckets['cpp'].append(script)
                    else:
                        buckets['c'].append(script)

        return buckets
    
    def annotate(self, codebase, langbase=None, isAI=False): 

        langgroups = {}
        if not langbase:
            langgroups = self.groupby_lang(codebase)
        else:
            langgroups = {x: y for (x, y) in zip(langbase, codebase)}
        dataframes = {}

        for lang, scripts in langgroups.items(): 
            
            df = self.codecorpus.frame(scripts, lang, isAI)
            dataframes[lang] = df

        master_frame = pd.concat([frame for lang, frame in dataframes.items()], ignore_index=True)

        return master_frame

    def saveas(self, df, name, path="./"): 

        path = Path(path) 
        if not os.path.exists(path): 
            path.mkdir(parents=True, exist_ok=True)
        
        filepath = path / name

        df.to_csv(filepath, index=False)

    def loadfrom(self, filename, columns=[], isAI=False):
        df = []
        if filename.endswith('.jsonl'):
            df = pd.read_json(filename, lines=True)
        else:
            df = pd.read_csv(filename)
        
        if columns:
            relevant_columns = df[columns]
            return relevant_columns

        return columns




        








        
            









