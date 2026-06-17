import asyncio 
from ollama import AsyncClient
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import json
import os
import aiofiles


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CODE_DIR = DATA_DIR / "code"
LLM_DIR = DATA_DIR / "llm_code" 
PROMPT_DIR = CODE_DIR / "promptfiles"

class ollama_async:
    def __init__(self, workers=4):
        self.model = "qwen2.5-coder:1.5b"
        self.num_workers = workers
        self.client = AsyncClient()
        self.queue = asyncio.Queue()
        self.results: List[Optional[str]] = []

        self.folder = PROMPT_DIR / "summary_prompts"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.folder / "data_backup.jsonl"
    
    def load_docs(self, filename, column ="src"):
        df = pd.read_csv(filename)
        docs = df[column].dropna().astype(str).tolist()

        return docs

    def save_csv(self, responses, saveas):
        indexed_responses = {
                "doc_id": list(range(1, len(responses)+1)),
                "model_response": responses
                }

        
        folder = self.folder
        folder.mkdir(parents=True, exist_ok=True)
        filename = saveas+".csv" if not saveas.endswith(".csv") else saveas

        path = folder / filename
        
        df = pd.DataFrame(indexed_responses)
        df.to_csv(path, index=False)
        print(f"saved successfully to {path}")

    def load_checkpoint(self) -> set:
        processed = set()
        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        processed.add(data["doc_id"])
                    except json.JSONDecodeError:
                        continue
        return processed

    def json_to_csv(self, csvname): 

        filename = csvname + ".csv" if not csvname.endswith(".csv") else csvname
        path = self.folder / filename

        df = pd.read_json(self.jsonl_path, lines=True)
        df = df.sort_values(by="doc_id")
        df.to_csv(path, index=False)

        print(f"Final complete model resonse csv saved to {str(path)}")


    async def manager(self, _worker, queue, docs, progress, file_lock):
        system_instruction = "Output ONLY a prose summary of the code formatted for prompting an ai. No code. No chatter. No intro phrases. No markdown formatting. No commentary."

        while True:
            i = await queue.get()
            if i is None:
                queue.task_done()
                break

            doc_id = i+1

            cur_doc = docs[i]
            doc_words = cur_doc.split()
            sample = doc_words[:200]
            sample_str = " ".join(sample)
            instruction = f"Generate a short prompt based on this code to prompt an LLM to write a similar program: \n{sample_str}"

            try:
                response = await self.client.generate(
                                    model=self.model, 
                                    prompt=instruction,
                                    system=system_instruction,
                                    options={
                                        "temperature": 0.4,
                                        "top_p": 0.8
                                        }
                                    )
                strip_response = response['response'].strip()

                row = {"doc_id": doc_id, "model_response": strip_response}

                async with file_lock:
                    async with aiofiles.open(self.jsonl_path, mode="a") as f:
                        await f.write(json.dumps(row) + '\n')

            except Exception as e:
                error_row = {"doc_id": doc_id, "model_response": f"ERROR: {str(e)}"}
                async with file_lock:
                    async with aiofiles.open(self.jsonl_path, mode="a") as f:
                        await f.write(json.dumps(error_row) + '\n')

            finally:
                progress.update(1)
                queue.task_done()
        
    async def main(self, infile, saveas="model_generated_promts"):

        docs = self.load_docs(infile, "src")
        self.results = [None] * len(docs)
        queue = self.queue

        file_lock = asyncio.Lock()

        completed_indexes = self.load_checkpoint()
        if completed_indexes: 
            print(f"Previous checkpoint found, processing from document: {len(completed_indexes)}")

        q_count = 0

        for i in range(len(docs)):
            doc_id = i+1
            if doc_id in completed_indexes: 
                continue
            await queue.put(i)
            q_count += 1

        if q_count == 0: 
            print("all documents processed") 
            self.json_to_csv(saveas)
            return 


        for _ in range(self.num_workers):
            await queue.put(None)

        progbar = tqdm_asyncio(total=q_count, desc="Generating code summary prompts")

        workers = []

        for workerID in range(self.num_workers):
            task = asyncio.create_task(self.manager(workerID, queue, docs, progbar, file_lock))
            workers.append(task)

        await asyncio.gather(*workers)

        progbar.close()

        print(f"\nProcessed {len(docs)} and created prompts ")
        self.json_to_csv(saveas)

if __name__ == "__main__": 

    processor = ollama_async(workers=1)

    csv = CODE_DIR / "new_webscraped_code.csv"

    asyncio.run(processor.main(infile=csv))






    






