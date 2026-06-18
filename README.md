## Lexi 

### LLM/Human language corpora and code datasets with multidimensional data analysis

A tool to build and analyze topic matched corpora of LLM and Human **prose** and **code** for training classification models

### PROSE data 

#### Key Metrics Included:

* `text`: The raw text sample.

* `pos`: Part-of-Speech tags.

* `lemmas`: Lemmatized tokens.

* `mean sentence length`: Avg tokens per sentence.

* `std sentence length`: Standard Deviation of tokens per sentence.

* `lexical density`: Ratio of content words to total words. 

**Lexical Density**

$$\frac{\lVert CONTW \rVert}{\lVert W \rVert}$$

* `variance`: Token frequency variance across document.

* `burstiness`: Ratio of hapaxes to total tokens.

**Burstiness**

$$\frac{HAPX}{\lVert TOK \rVert}$$

* `saliency`: Sum of token frequencies divided by number of unique tokens (types).

**Saliency**

$$\frac{\sum_{w} FDIST}{\lVert TYPES \rVert}$$

* `sentiment`: Sentiment polarity score.

* `sentiment deviation`: Absolute value of sentence sentiment minus document sentiment, divided by number of sentences. 

**Sentiment Deviation**

$$\frac{|S.SENTIMTENT - D.SENTIMENT|}{\lVert S \rVert}$$

* `ttr`: Type-Token Ratio (lexical diversity).

* `depth`: Max depth of syntax trees. 

* `head`: Sum over sentences of head locations, divided by number of sentences.

**Sentence Head Location**

$$\frac{\sum_{s} HEAD.LOC}{\lVert S \rVert}$$ 

* `sub clauses`: Avg number of suboordinate clauses per sentence.

* `coord clauses`: Avg number of coordinating conjunctions per sentence.

* `branching` : Left/right bias of branching in syntax trees across document.

* `balanced`: Boolean, Left branch nodes == Right branch nodes. (+/- 2 nodes).

#### Dataset Sample
<div style="overflow-x: auto; max-width: 100%;">

<details>
<summary><b>▶ Topic 0 Comparison</b></summary>

|topic_id|source|text|pos|lemmas|mean sentence length|std sent len|lex-density|var|burstiness|saliency|sentiment|sent sd|ttr|depth|head|subord|coord|branch|blnc|label|tmstmp |
|---:|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
|0| **Human** | Sentiment. Tension management and communication of... | ['NN', '.', 'NNP', '... | ['Sentiment', '.', '... | 22.3137 | 15.4457 | 0.516257 | 5.60993e-06 | 1.67315 | 0.00141|0.996|0.31|0.297|15|0.022|0.509| 0.823|0.135|0.5|HUMAN|2026-06-01_085540|
|0| **LLM** | The Architecture of Belonging: Tradition, Boundary... | ['DT', 'NNP', 'IN', ... | ['The', 'Architectur... | 22.4915 | 9.84864 | 0.522984 | 4.75066e-06 | 1.17536 | 0.0018|0.99|0.325|0.39|13|0.018|0.84|0.694|-0.002|0.37|LLM|2026-06-01_110934|

</details>

<details>
<summary><b>▶ Topic 1 Comparison</b></summary>

|topic_id|source|text|pos|lemmas|mean sentence length|std sent len|lex-density|var|burstiness|saliency|sentiment|sent sd|ttr|depth|head|subord|coord|branch|blnc|label|tmstmp |
|---:|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
|1| **Human** | When Fred wheeled him back into his room, the big ... | ['WRB', 'NNP', 'VBD'... | ['When', 'Fred', 'wh... | 23.0693 | 13.8301 | 0.497854 | 2.15642e-06 | 1.30916 | 0.001|0.99|0.407|0.309|12|0.025|1.267|0.9|0.008|0.3|HUMAN|2026-06-01_085541|
|1|**LLM**|The following narrative expands upon the fragmente... |['DT', 'JJ', 'JJ', '... |['The', 'following',... |22.35|12.23|0.47|2.6e-06|1.58|0.001|0.99|0.35|0.26|14|0.023|1.05|0.66|0.011|0.48|LLM|2026-06-01_110935|

</details>

<details>
<summary><b>▶ Topic 2 Comparison</b></summary>

|topic_id|source|text|pos|lemmas|mean sentence length|std sent len|lex-density|var|burstiness|saliency|sentiment|sent sd|ttr|depth|head|subord|coord|branch|blnc|label|tmstmp |
|---:|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
|2|**Human**|In good time I shall get to the distressing actual... |['IN', 'JJ', 'NN', '... |['In', 'good', 'time...|17.7388|11.5848|0.459823|1.71e-06|1.21|0.001|-0.9|0.34|0.32|10|0.03|0.88|0.57|0.006|0.54|HUMAN|2026-06-01_085541|
|2|**LLM**|To understand the character of Salu Norberg—not me...|['TO', 'VB', 'DT', '...|['To', 'understand',...|19.15|11.85|0.45|5.8e-06|1.43|0.001|0.9|0.21|0.32|12|0.023|1|0.45|-0.017|0.53|LLM|2026-06-01_110936|

</details>

</div>

#### Files 

```text
src/
├── classifier.py
├── code_crawler.cpp
├── dataset_annotator.py
├── dataset_builder.py
├── encorporate.py
├── ensemble_classifier.py
├── gemini.py
├── gemini_async.py
├── main.py
├── ollama_async.py
└── train_code_classifier.py
models/
├── code_models
│   ├── c_cpp_expert.joblib
│   ├── c_cpp_scaler.joblib
│   ├── c_expert.joblib
│   ├── c_scaler.joblib
│   ├── cpp_expert.joblib
│   ├── cpp_scaler.joblib
│   ├── java_expert.joblib
│   ├── java_scaler.joblib
│   ├── language_router.joblib
│   ├── py_expert.joblib
│   ├── py_scaler.joblib
│   └── vectorizerNB.joblib
└── prose_models
    └── paired_data_ensemble_model.joblib
```
#### 
