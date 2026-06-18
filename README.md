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

**LEXICAL DENSITY**$$=\frac{||CONTW||}{||W||}$$

* `variance`: Token frequency variance across document.

* `burstiness`: Ratio of hapaxes to total tokens.

**BURSTINESS**$$= \frac{HAPX}{||TOK||}$$

* `saliency`: Sum of token frequencies divided by number of unique tokens (types).

**SALIENCY**$$=\frac{\sum_{w} FDIST}{||TYPES||}$$

* `sentiment`: Sentiment polarity score.

* `sentiment deviation`: Absolute value of sentence sentiment minus document sentiment, divided by number of sentences. 

**SENTIMENT DEVIATION**$$= \frac{|S.SENTIMTENT - D.SENTIMENT|}{||S||}$$

* `ttr`: Type-Token Ratio (lexical diversity).

* `depth`: Max depth of syntax trees. 

* `head`: Sum over sentences of head locations, divided by number of sentences.

**SENTENCE HEAD LOC**$$=\frac{\sum_{s} HEAD.LOC}{||S||}$$ 

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
| 0 | **Human** | Sentiment. Tension management and communication of... | ['NN', '.', 'NNP', '... | ['Sentiment', '.', '... | 22.3137 | 15.4457 | 0.516257 | 5.60993e-06 | 1.67315 | 0.00141561 | 0.9969 | 0.312041 | 0.297452 | 15 | 0.0228189 | 0.509804 | 0.823529 | 0.135929 | 0.5 | HUMAN | 2026-06-01_085540 |
| 0 | **LLM** | The Architecture of Belonging: Tradition, Boundary... | ['DT', 'NNP', 'IN', ... | ['The', 'Architectur... | 22.4915 | 9.84864 | 0.522984 | 4.75066e-06 | 1.17536 | 0.00185441 | 0.9962 | 0.325832 | 0.394876 | 13 | 0.0188296 | 0.847458 | 0.694915 | -0.00201933 | 0.372881 | LLM | 2026-06-01_110934 |

</details>

<details>
<summary><b>▶ Topic 1 Comparison</b></summary>

|topic_id|source|text|pos|lemmas|mean sentence length|std sent len|lex-density|var|burstiness|saliency|sentiment|sent sd|ttr|depth|head|subord|coord|branch|blnc|label|tmstmp |
|---:|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| 1 | **Human** | When Fred wheeled him back into his room, the big ... | ['WRB', 'NNP', 'VBD'... | ['When', 'Fred', 'wh... | 23.0693 | 13.8301 | 0.497854 | 2.15642e-06 | 1.30916 | 0.0011217 | 0.9983 | 0.40705 | 0.309871 | 12 | 0.0250197 | 1.26733 | 0.90099 | 0.00801188 | 0.356436 | HUMAN | 2026-06-01_085541 |
| 1 | **LLM** | The following narrative expands upon the fragmente... | ['DT', 'JJ', 'JJ', '... | ['The', 'following',... | 22.3506 | 12.2382 | 0.470947 | 2.6522e-06 | 1.58826 | 0.00102537 | 0.9988 | 0.355442 | 0.26932 | 14 | 0.0238958 | 1.05844 | 0.668831 | 0.0117418 | 0.480519 | LLM | 2026-06-01_110935 |

</details>

<details>
<summary><b>▶ Topic 2 Comparison</b></summary>

|topic_id|source|text|pos|lemmas|mean sentence length|std sent len|lex-density|var|burstiness|saliency|sentiment|sent sd|ttr|depth|head|subord|coord|branch|blnc|label|tmstmp |
|---:|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| 2 | **Human** | In good time I shall get to the distressing actual... | ['IN', 'JJ', 'NN', '... | ['In', 'good', 'time... | 17.7388 | 11.5848 | 0.459823 | 1.75091e-06 | 1.21884 | 0.00108564 | -0.9852 | 0.343956 | 0.327303 | 10 | 0.0336383 | 0.880597 | 0.574627 | 0.00659567 | 0.544776 | HUMAN | 2026-06-01_085541 |
| 2 | **LLM** | To understand the character of Salu Norberg—not me... | ['TO', 'VB', 'DT', '... | ['To', 'understand',... | 19.1538 | 11.8544 | 0.452668 | 5.81684e-06 | 1.43173 | 0.00168455 | 0.9987 | 0.214917 | 0.321285 | 12 | 0.0237924 | 1 | 0.450549 | -0.0179107 | 0.538462 | LLM | 2026-06-01_110936 |

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
