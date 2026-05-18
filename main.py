from dataset_builder import dataset_builder, dataset_builder_ai, data_generator

import pandas as pd 
import numpy as np 
import os 




def menu():

    print("Main Menu: ")
    print("1. Create LLM and Human PROSE data")
    print("2. Create LLM and Human CODE data")
    print("3. Convert file(s) to annotated data frame")

    choice = input("\nEnter your choice, or enter 'x' to exit: ")

    return choice

def cli():

    annotator = data_generator()

    while (True): 

        x = menu()

        if x.lower() == 'x':
            print("Exiting...")
            return 

        elif x == 1: 

            num_topics = input("\nEnter number of topics: ")
            num_calls = input("\nEnter number of calls to model per topic: ")
            label = input("\nEnter name for dataset: ")
            sysprompts = input("\nDo you want to customize the system prompt for one or both of the LLM bots?[Y/n]: ")
            bots = []
            if sysprompts.lower == 'Y': 
                bot_a = input("\nEnter system prompt for bot one: ")
                bot_b = input("\nEnter system prompt for bot two: ")
                bots.append(bot_a)
                bots.append(bot_b)
            
            bot_one = ""
            bot_two = ""
            if len(bots) > 1: 
                bot_one = bots[0]
                bot_two = bots[1]
            elif len(bots) > 0:
                bot_one = bots[0]
                bot_two = None
            else:
                bot_one = None
                bot_two = None

            ai_prose, human_prose = annotator.generate_prose(name=label, sys_a=bot_one, sys_b=bot_two, no_topics=int(num_topics), no_calls=int(num_calls))

        elif x == 2:
            
            num_rows = input("\nEnter number of code instances to be created per dataset (ai and human): ")
            label = input("\nEnter name for dataset: ")

            ai_code, human_code = annotator.generate_code(name=label, rows=int(num_rows))

        elif x == 3: 

            code_files = []
            code_bools = []
            prose_files = []
            prose_bools = []

            filetypes = input("\nEnter [1] for prose file(s), [2] for code file(s), [3] for both: ")

            if filetypes == "1" or filetypes == "3":
                print("Enter PROSE filenames (.txt or equivalent files only) to annotate, when finished enter 'x' to continue")
                while True: 
                    prosefile = input("\nEnter filename: ")
                    if prosefile.lower() == 'x': 
                        break
                    ai = input("\nIs the file AI generated? [Y/n]: ")
                    isAI = True if ai.lower() == 'y' else False
                    prose_files.append(prosefile)
                    prose_bools.append(isAI)
            
            if filetypes == "2" or filetypes == "3": 
                print("Enter CODE filenames (.txt, .py, .cpp, .c, or .java) to annotate, when finished enter 'x' to continue")
                while True:
                    codefile = input("\nEnter filename: ")
                    if codefile.lower() == 'x':
                        break
                    ai = input("\nIs the file AI generated? [Y/n]: ")
                    isAI = True if ai.lower() == 'y' else False
                    code_files.append(codefile)
                    code_bools.append(isAI)

            for file, boolean in zip(prose_files, prose_bools):
                annotated_file = annotator.annotate_prose_file(file, boolean)
                if annotated_file:
                    print(f"{file} annotated and saved as {annotated_file}")

            for file, boolean in zip(code_files, code_bools):
                annotated_code = annotator.annotate_code_file(file, boolean)
                if annotated_code:
                    print(f"{file} annotated and saved as {annotated_code}")

        else:
            print("Invalid input, please choose an option from the list")
    


print(f"~~~~~~~~~~~~~~~~PARSE LLM~~~~~~~~~~~~~~~~~~~")
print(f"~~~~~custom LLM and NLP dataset builder~~~~~")

cli()

print("Goodbye")

