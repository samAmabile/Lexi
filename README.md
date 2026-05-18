# Lexi Frame LLM and Natural Language Dataframe Compiler 
## Streamlined generation of LLM and human language and code

### Modules:

### gemini.py

* Chat and Automate:

To Use:
``` 
from geimini import Chat, Automate 
```
Offers both manual *(Chat)* and automated *(Automate)* chat with Gemini models, with automatic capture of history in a csv. 

* Code:

To Use:

```
from gemini import Code
```
Offers both manual and automated generation of single code instances (one prompt) or large codebases (from default prompt lists or generated prompts)

Automatically saves sessions to a dataframe. 

### encorporate.py

* Encorporator:

To Use:
```
from encorporate import Encorporator
```
Builds dataframe corpora from text (string) inputs with lexical, syntactic, and semantic data. 

![Encorporator dataframe](images/encorporator_dataframe.png)

* Codecorpus:

To Use:
```
from encorporate import Codecorps
```
Builds dataframes of code with detailed analysis from *TreeSitter* and *Lizard* libraries

![Codecorpus dataframe](images/Codecorpus_dataframe.png)

### dataset\_builder.py 

Leverages *gemini.py* and *encorporate.py* along with **NLTK** and **Hugging Face** to build paired datasets of LLM/human content for both prose and code

To Use:
```
from dataset_builder import data_generator
```


