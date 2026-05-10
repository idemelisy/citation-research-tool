# SRG Lite reading list

- Generated: `2026-05-10T19:23:19.070849`
- Query profile: intent=`method`, terms=directed, preference, optimization
- Evidence quality: **HIGH** (references=1718, edges=209)
- Quality (dedup): merged_pairs=2, conflicts=0
- Source mix: arxiv=4, openalex=63, pending=8

## Reading guide

_Importance rank uses graph relevance, connectivity, and missing-link evidence. Use this as a practical reading order, not a ground-truth citation count._

- #1 **Strengthening Multimodal Large Language Model with Bootstrapped Preference Optimization** (`openalex:W4403726885`) — importance=5.380, relevance=0.990, graph_links=34
- #2 **Learning to summarize from human feedback** (`arxiv:2009.01325`) — importance=5.373, relevance=0.887, graph_links=36
- #3 **RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback** (`openalex:W4402727405`) — importance=3.200, relevance=1.000, graph_links=12
- #4 **A Deep Reinforced Model for Abstractive Summarization** (`openalex:W2612675303`) — importance=2.867, relevance=0.534, graph_links=18
- #5 **A bird's-eye view on coherence, and a worm's-eye view on cohesion.** (`openalex:W2899485151`) — importance=2.724, relevance=0.512, graph_links=17
- #6 **An Actor-Critic Algorithm for Sequence Prediction** (`openalex:W2487501366`) — importance=2.677, relevance=0.489, graph_links=17
- #7 **Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding** (`openalex:W4402753774`) — importance=2.459, relevance=0.630, graph_links=12
- #8 **Evaluating Object Hallucination in Large Vision-Language Models** (`openalex:W4389523832`) — importance=2.316, relevance=0.708, graph_links=9
- #9 **The Natural Language Decathlon: Multitask Learning as Question Answering** (`openalex:W2809324505`) — importance=2.263, relevance=0.431, graph_links=14
- #10 **ShareGPT4V: Improving Large Multi-modal Models with Better Captions** (`openalex:W4404575065`) — importance=2.260, relevance=0.730, graph_links=8

## Topic separation

### General Preference Alignment

- **Learning to summarize from human feedback** (`arxiv:2009.01325`) (importance=5.373)
- **RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback** (`openalex:W4402727405`) (importance=3.200)
- **A Deep Reinforced Model for Abstractive Summarization** (`openalex:W2612675303`) (importance=2.867)
- **A bird's-eye view on coherence, and a worm's-eye view on cohesion.** (`openalex:W2899485151`) (importance=2.724)
- **An Actor-Critic Algorithm for Sequence Prediction** (`openalex:W2487501366`) (importance=2.677)
- **The Natural Language Decathlon: Multitask Learning as Question Answering** (`openalex:W2809324505`) (importance=2.263)
- **ShareGPT4V: Improving Large Multi-modal Models with Better Captions** (`openalex:W4404575065`) (importance=2.260)
- **Sequence Level Training with Recurrent Neural Networks** (`openalex:W2176263492`) (importance=2.051)
- _... 47 more in this topic_

### Multimodal / Vision

- **Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding** (`openalex:W4402753774`) (importance=2.459)
- **Evaluating Object Hallucination in Large Vision-Language Models** (`openalex:W4389523832`) (importance=2.316)
- **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning** (`openalex:W4376312115`) (importance=1.707)
- **LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding** (`openalex:W4382766522`) (importance=0.809)
- **The Instinctive Bias: Spurious Images lead to Illusion in MLLMs** (`openalex:W4404782342`) (importance=0.809)
- **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond** (`openalex:W4386185600`) (importance=0.756)

### Core DPO Methods

- **Strengthening Multimodal Large Language Model with Bootstrapped Preference Optimization** (`openalex:W4403726885`) (importance=5.380)
- **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (`openalex:W4378771755`) (importance=0.721)
- **Iterative Reasoning Preference Optimization** (`arxiv:2404.19733`) (importance=0.201)
- **ORPO: Monolithic Preference Optimization without Reference Model** (`arxiv:2403.07691`) (importance=0.201)

### Data Selection / Curation

- **Understanding the difficulty of training deep feedforward neural networks** (`openalex:W1533861849`) (importance=0.387)

### Reasoning / Stepwise

- **DetGPT: Detect What You Need via Reasoning** (`openalex:W4389519620`) (importance=1.832)

## Reference coverage

- Papers with explicit `references` field: **42 / 67**
- Supported citation edges in graph: **90** (potential/low-confidence: **119**)
- Per-paper reference counts (top 15 by importance):
  - `openalex:W4403726885`: 34 references
  - `arxiv:2009.01325`: 71 references
  - `openalex:W4402727405`: 76 references
  - `openalex:W2612675303`: 39 references
  - `openalex:W2899485151`: 55 references
  - `openalex:W2487501366`: 40 references
  - `openalex:W4402753774`: 78 references
  - `openalex:W4389523832`: 37 references
  - `openalex:W2809324505`: 106 references
  - `openalex:W4404575065`: 39 references
  - `openalex:W2176263492`: 29 references
  - `openalex:W4387947626`: 44 references
  - `openalex:W4385573325`: 66 references
  - `openalex:W1843891098`: 24 references
  - `openalex:W4389519620`: 55 references

## Foundational papers

### Highly ranked in this graph

_Bright gold (foundational) hubs — pinned at graph center in the UI._

- ★ **Strengthening Multimodal Large Language Model with Bootstrapped Preference Optimization** — PII concept match ≈0.0, graph degree 34 (`openalex:W4403726885`)
- ★ **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** — PII concept match ≈0.0, graph degree 1 (`openalex:W4378771755`)

- **Strengthening Multimodal Large Language Model with Bootstrapped Preference Optimization** — score 2.16: _sqrt(degree) × recency + metadata completeness (hub-resistant)_ (`openalex:W4403726885`)
- **Pending metadata: W2962735233** — score 1.94: _sqrt(degree) × recency + metadata completeness (hub-resistant)_ (`openalex:W2962735233`)
- **Pending metadata: W6778883912** — score 1.752: _sqrt(degree) × recency + metadata completeness (hub-resistant)_ (`openalex:W6778883912`)
- **Pending metadata: W2322584079** — score 1.53: _sqrt(degree) × recency + metadata completeness (hub-resistant)_ (`openalex:W2322584079`)
- **Learning to summarize from human feedback** — score 1.274: _sqrt(degree) × recency + metadata completeness (hub-resistant)_ (`arxiv:2009.01325`)
- **Pending metadata: W6600384961** — score 1.24: _sqrt(degree) × recency + metadata completeness (hub-resistant)_ (`openalex:W6600384961`)
- **Pending metadata: W6607367167** — score 1.24: _sqrt(degree) × recency + metadata completeness (hub-resistant)_ (`openalex:W6607367167`)
- **Pending metadata: W6679942958** — score 1.24: _sqrt(degree) × recency + metadata completeness (hub-resistant)_ (`openalex:W6679942958`)
- **Pending metadata: W6777574746** — score 1.24: _sqrt(degree) × recency + metadata completeness (hub-resistant)_ (`openalex:W6777574746`)
- **Pending metadata: W6782465632** — score 1.24: _sqrt(degree) × recency + metadata completeness (hub-resistant)_ (`openalex:W6782465632`)

## Missing link candidates (foundational PII literature)

_None above threshold._


## Trends (domain-aware)

### Publication years

- 1952: 1
- 1989: 1
- 1994: 1
- 2002: 1
- 2003: 1
- 2004: 1
- 2005: 1
- 2010: 2
- 2014: 1
- 2015: 4
- 2016: 5
- 2017: 6
- 2018: 7
- 2019: 3
- 2020: 1
- 2022: 4
- 2023: 15
- 2024: 12
- unknown: 8

### Domains

- cs: 64
- unknown: 8
- biomedical: 2
- physics: 1


## Supported citation links (medium / high confidence)

- `arxiv:2009.01325` → `openalex:W1843891098` (medium) _Learning to summarize from human feedback_ → _A Neural Attention Model for Abstractive Sentence Summarizat_
  - Snippet: _Abstract mentions 'a'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W1973435495` (medium) _Learning to summarize from human feedback_ → _Learning to rank for information retrieval_
  - Snippet: _Abstract mentions 'learning'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W2047221353` (medium) _Learning to summarize from human feedback_ → _Optimizing search engines using clickthrough data_
  - Snippet: _Abstract mentions 'optimizing'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W2487501366` (medium) _Learning to summarize from human feedback_ → _An Actor-Critic Algorithm for Sequence Prediction_
  - Snippet: _Abstract mentions 'an'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W2564034046` (medium) _Learning to summarize from human feedback_ → _Tuning Recurrent Neural Networks with Reinforcement Learning_
  - Snippet: _Abstract mentions 'tuning'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W2612675303` (medium) _Learning to summarize from human feedback_ → _A Deep Reinforced Model for Abstractive Summarization_
  - Snippet: _Abstract mentions 'a'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W2737114849` (medium) _Learning to summarize from human feedback_ → _Reinforcement Learning for Bandit Neural Machine Translation_
  - Snippet: _Abstract mentions 'reinforcement'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W2741672218` (medium) _Learning to summarize from human feedback_ → _The limits of automatic summarisation according to ROUGE_
  - Snippet: _Abstract mentions 'the'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W2798047776` (medium) _Learning to summarize from human feedback_ → _Can Neural Machine Translation be Improved with User Feedbac_
  - Snippet: _Abstract mentions 'can'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W2809324505` (medium) _Learning to summarize from human feedback_ → _The Natural Language Decathlon: Multitask Learning as Questi_
  - Snippet: _Abstract mentions 'the'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W2899485151` (medium) _Learning to summarize from human feedback_ → _A bird's-eye view on coherence, and a worm's-eye view on coh_
  - Snippet: _Abstract mentions 'a'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W2910458567` (medium) _Learning to summarize from human feedback_ → _Learning from Dialogue after Deployment: Feed Yourself, Chat_
  - Snippet: _Abstract mentions 'learning'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W2938704169` (medium) _Learning to summarize from human feedback_ → _The Curious Case of Neural Text Degeneration_
  - Snippet: _Abstract mentions 'the'_ (`abstract_proxy`)
- `arxiv:2009.01325` → `openalex:W4385573325` (high) _Learning to summarize from human feedback_ → _ZeroGen: Efficient Zero-shot Learning via Dataset Generation_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=3)_ (`bibliographic_coupling`)
- `openalex:W1522301498` → `openalex:W2170973209` (high) _Adam: A Method for Stochastic Optimization_ → _Semi-supervised Sequence Learning_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W1544827683` → `openalex:W1843891098` (high) _Teaching Machines to Read and Comprehend_ → _A Neural Attention Model for Abstractive Sentence Summarizat_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W1544827683` → `openalex:W2176263492` (high) _Teaching Machines to Read and Comprehend_ → _Sequence Level Training with Recurrent Neural Networks_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=3)_ (`bibliographic_coupling`)
- `openalex:W1544827683` → `openalex:W2487501366` (high) _Teaching Machines to Read and Comprehend_ → _An Actor-Critic Algorithm for Sequence Prediction_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=3)_ (`bibliographic_coupling`)
- `openalex:W1544827683` → `openalex:W2612675303` (high) _Teaching Machines to Read and Comprehend_ → _A Deep Reinforced Model for Abstractive Summarization_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=3)_ (`bibliographic_coupling`)
- `openalex:W1544827683` → `openalex:W2798299685` (high) _Teaching Machines to Read and Comprehend_ → _Improving a Neural Semantic Parser by Counterfactual Learnin_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W1544827683` → `openalex:W2809324505` (high) _Teaching Machines to Read and Comprehend_ → _The Natural Language Decathlon: Multitask Learning as Questi_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=4)_ (`bibliographic_coupling`)
- `openalex:W1544827683` → `openalex:W2899485151` (high) _Teaching Machines to Read and Comprehend_ → _A bird's-eye view on coherence, and a worm's-eye view on coh_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=3)_ (`bibliographic_coupling`)
- `openalex:W1544827683` → `openalex:W2941610736` (high) _Teaching Machines to Read and Comprehend_ → _Abstract Text Summarization with a Convolutional Seq2seq Mod_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W1843891098` → `openalex:W2467173223` (high) _A Neural Attention Model for Abstractive Sentence Summarizat_ → _Abstractive Sentence Summarization with Attentive Recurrent _
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=11)_ (`bibliographic_coupling`)
- `openalex:W1843891098` → `openalex:W2798047776` (high) _A Neural Attention Model for Abstractive Sentence Summarizat_ → _Can Neural Machine Translation be Improved with User Feedbac_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W1843891098` → `openalex:W2798299685` (high) _A Neural Attention Model for Abstractive Sentence Summarizat_ → _Improving a Neural Semantic Parser by Counterfactual Learnin_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W1843891098` → `openalex:W2809324505` (high) _A Neural Attention Model for Abstractive Sentence Summarizat_ → _The Natural Language Decathlon: Multitask Learning as Questi_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W1843891098` → `openalex:W2899485151` (high) _A Neural Attention Model for Abstractive Sentence Summarizat_ → _A bird's-eye view on coherence, and a worm's-eye view on coh_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W1843891098` → `openalex:W2941610736` (high) _A Neural Attention Model for Abstractive Sentence Summarizat_ → _Abstract Text Summarization with a Convolutional Seq2seq Mod_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W1973435495` → `openalex:W2152314154` (high) _Learning to rank for information retrieval_ → _Accurately interpreting clickthrough data as implicit feedba_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=4)_ (`bibliographic_coupling`)
- `openalex:W2081265723` → `openalex:W2899485151` (high) _Hedge Trimmer_ → _A bird's-eye view on coherence, and a worm's-eye view on coh_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W2108325777` → `openalex:W2487501366` (high) _Automatic evaluation of machine translation quality using lo_ → _An Actor-Critic Algorithm for Sequence Prediction_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W2108325777` → `openalex:W2741672218` (high) _Automatic evaluation of machine translation quality using lo_ → _The limits of automatic summarisation according to ROUGE_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W2108325777` → `openalex:W2809324505` (high) _Automatic evaluation of machine translation quality using lo_ → _The Natural Language Decathlon: Multitask Learning as Questi_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W2108325777` → `openalex:W2899485151` (high) _Automatic evaluation of machine translation quality using lo_ → _A bird's-eye view on coherence, and a worm's-eye view on coh_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W2170973209` → `openalex:W2176263492` (high) _Semi-supervised Sequence Learning_ → _Sequence Level Training with Recurrent Neural Networks_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W2170973209` → `openalex:W2487501366` (high) _Semi-supervised Sequence Learning_ → _An Actor-Critic Algorithm for Sequence Prediction_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)
- `openalex:W2170973209` → `openalex:W2809324505` (high) _Semi-supervised Sequence Learning_ → _The Natural Language Decathlon: Multitask Learning as Questi_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=3)_ (`bibliographic_coupling`)
- `openalex:W2170973209` → `openalex:W2899485151` (high) _Semi-supervised Sequence Learning_ → _A bird's-eye view on coherence, and a worm's-eye view on coh_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=3)_ (`bibliographic_coupling`)
- `openalex:W2170973209` → `openalex:W4385573325` (high) _Semi-supervised Sequence Learning_ → _ZeroGen: Efficient Zero-shot Learning via Dataset Generation_
  - Snippet: _Strong Evidence: bibliographic coupling (shared references=2)_ (`bibliographic_coupling`)

## Potential connections (lower confidence)

_Dashed / low-confidence edges: abstract proxy or unresolved context; treat as exploratory._

- `openalex:W4403726885` → `openalex:W2013784666` (low) _Strengthening Multimodal Large Language Model with_ → _RANK ANALYSIS OF INCOMPLETE BLOCK DESIGNS_
- `openalex:W4403726885` → `openalex:W2962735233` (low) _Strengthening Multimodal Large Language Model with_ → _Pending metadata: W2962735233_
- `openalex:W4403726885` → `openalex:W4226278401` (low) _Strengthening Multimodal Large Language Model with_ → _Training language models to follow instructions wi_
- `openalex:W4403726885` → `openalex:W4281790610` (low) _Strengthening Multimodal Large Language Model with_ → _Self-Guided Noise-Free Data Generation for Efficie_
- `openalex:W4403726885` → `openalex:W4366196653` (low) _Strengthening Multimodal Large Language Model with_ → _RAFT: Reward rAnked FineTuning for Generative Foun_
- `openalex:W4403726885` → `openalex:W4367628410` (low) _Strengthening Multimodal Large Language Model with_ → _LLaMA-Adapter V2: Parameter-Efficient Visual Instr_
- `openalex:W4403726885` → `openalex:W4376312115` (low) _Strengthening Multimodal Large Language Model with_ → _InstructBLIP: Towards General-purpose Vision-Langu_
- `openalex:W4403726885` → `openalex:W4379259189` (low) _Strengthening Multimodal Large Language Model with_ → _LLaVA-Med: Training a Large Language-and-Vision As_
- `openalex:W4403726885` → `openalex:W4382766522` (low) _Strengthening Multimodal Large Language Model with_ → _LLaVAR: Enhanced Visual Instruction Tuning for Tex_
- `openalex:W4403726885` → `openalex:W4384817778` (low) _Strengthening Multimodal Large Language Model with_ → _Mathematical Analysis of Machine Learning Algorith_
- `openalex:W4403726885` → `openalex:W4385573325` (low) _Strengthening Multimodal Large Language Model with_ → _ZeroGen: Efficient Zero-shot Learning via Dataset _
- `openalex:W4403726885` → `openalex:W4385645323` (low) _Strengthening Multimodal Large Language Model with_ → _MM-Vet: Evaluating Large Multimodal Models for Int_
- `openalex:W4403726885` → `openalex:W4386185600` (low) _Strengthening Multimodal Large Language Model with_ → _Qwen-VL: A Versatile Vision-Language Model for Und_
- `openalex:W4403726885` → `openalex:W4386978002` (low) _Strengthening Multimodal Large Language Model with_ → _MetaMath: Bootstrap Your Own Mathematical Question_
- `openalex:W4403726885` → `openalex:W4387947626` (low) _Strengthening Multimodal Large Language Model with_ → _Woodpecker: Hallucination Correction for Multimoda_
- `openalex:W4403726885` → `openalex:W4389519620` (low) _Strengthening Multimodal Large Language Model with_ → _DetGPT: Detect What You Need via Reasoning_
- `openalex:W4403726885` → `openalex:W4389523832` (low) _Strengthening Multimodal Large Language Model with_ → _Evaluating Object Hallucination in Large Vision-La_
- `openalex:W4403726885` → `openalex:W4389977164` (low) _Strengthening Multimodal Large Language Model with_ → _G-LLaVA: Solving Geometric Problem with Multi-Moda_
- `openalex:W4403726885` → `openalex:W4389983382` (low) _Strengthening Multimodal Large Language Model with_ → _Silkie: Preference Distillation for Large Visual L_
- `openalex:W4403726885` → `openalex:W4401042360` (low) _Strengthening Multimodal Large Language Model with_ → _LMFlow: An Extensible Toolkit for Finetuning and I_
- `openalex:W4403726885` → `openalex:W4402670859` (low) _Strengthening Multimodal Large Language Model with_ → _Aligning Large Multimodal Models with Factually Au_
- `openalex:W4403726885` → `openalex:W4402671571` (low) _Strengthening Multimodal Large Language Model with_ → _Arithmetic Control of LLMs for Diverse User Prefer_
- `openalex:W4403726885` → `openalex:W4402727405` (low) _Strengthening Multimodal Large Language Model with_ → _RLHF-V: Towards Trustworthy MLLMs via Behavior Ali_
- `openalex:W4403726885` → `openalex:W4402753774` (low) _Strengthening Multimodal Large Language Model with_ → _Mitigating Object Hallucinations in Large Vision-L_
- `openalex:W4403726885` → `openalex:W4402781286` (low) _Strengthening Multimodal Large Language Model with_ → _PerceptionGPT: Effectively Fusing Visual Perceptio_
- `openalex:W4403726885` → `openalex:W4404575065` (low) _Strengthening Multimodal Large Language Model with_ → _ShareGPT4V: Improving Large Multi-modal Models wit_
- `openalex:W4403726885` → `openalex:W4404782342` (low) _Strengthening Multimodal Large Language Model with_ → _The Instinctive Bias: Spurious Images lead to Illu_
- `openalex:W4403726885` → `openalex:W4404783010` (low) _Strengthening Multimodal Large Language Model with_ → _MLLM-Protector: Ensuring MLLM’s Safety without Hur_
- `openalex:W4403726885` → `openalex:W6600384961` (low) _Strengthening Multimodal Large Language Model with_ → _Pending metadata: W6600384961_
- `openalex:W4403726885` → `openalex:W6607367167` (low) _Strengthening Multimodal Large Language Model with_ → _Pending metadata: W6607367167_
- `openalex:W4403726885` → `openalex:W6679942958` (low) _Strengthening Multimodal Large Language Model with_ → _Pending metadata: W6679942958_
- `openalex:W4403726885` → `openalex:W6777574746` (low) _Strengthening Multimodal Large Language Model with_ → _Pending metadata: W6777574746_
- `openalex:W4403726885` → `openalex:W6778883912` (low) _Strengthening Multimodal Large Language Model with_ → _Pending metadata: W6778883912_
- `openalex:W4403726885` → `openalex:W6782465632` (low) _Strengthening Multimodal Large Language Model with_ → _Pending metadata: W6782465632_
- `arxiv:2009.01325` → `openalex:W1522301498` (low) _Learning to summarize from human feedback_ → _Adam: A Method for Stochastic Optimization_
- `arxiv:2009.01325` → `openalex:W1533861849` (low) _Learning to summarize from human feedback_ → _Understanding the difficulty of training deep feed_
- `arxiv:2009.01325` → `openalex:W1544827683` (low) _Learning to summarize from human feedback_ → _Teaching Machines to Read and Comprehend_
- `arxiv:2009.01325` → `openalex:W2012318340` (low) _Learning to summarize from human feedback_ → _Optimum polynomial retrieval functions based on th_
- `arxiv:2009.01325` → `openalex:W2020237802` (low) _Learning to summarize from human feedback_ → _Automatic Combination of Multiple Ranked Retrieval_
- `arxiv:2009.01325` → `openalex:W2081265723` (low) _Learning to summarize from human feedback_ → _Hedge Trimmer_

## Node provenance (source labels)

- `openalex:W4378771755`: **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** — API
- `openalex:W4226278401`: **Training language models to follow instructions with human feedback** — API
- `openalex:W4403726885`: **Strengthening Multimodal Large Language Model with Bootstrapped Preference Optim** — API
- `arxiv:2404.19733`: **Iterative Reasoning Preference Optimization** — API
- `arxiv:2212.08073`: **Constitutional AI: Harmlessness from AI Feedback** — API
- `arxiv:2403.07691`: **ORPO: Monolithic Preference Optimization without Reference Model** — API
- `arxiv:2009.01325`: **Learning to summarize from human feedback** — API
- `openalex:W2013784666`: **RANK ANALYSIS OF INCOMPLETE BLOCK DESIGNS** — API
- `openalex:W4281790610`: **Self-Guided Noise-Free Data Generation for Efficient Zero-Shot Learning** — API
- `openalex:W4366196653`: **RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment** — API
- `openalex:W4367628410`: **LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model** — API
- `openalex:W4376312115`: **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tu** — API
- `openalex:W4379259189`: **LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One** — API
- `openalex:W4382766522`: **LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding** — API
- `openalex:W4384817778`: **Mathematical Analysis of Machine Learning Algorithms** — API
- `openalex:W4385573325`: **ZeroGen: Efficient Zero-shot Learning via Dataset Generation** — API
- `openalex:W4385645323`: **MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities** — API
- `openalex:W4386185600`: **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text** — API
- `openalex:W4386978002`: **MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models** — API
- `openalex:W4387947626`: **Woodpecker: Hallucination Correction for Multimodal Large Language Models** — API
- `openalex:W4389519620`: **DetGPT: Detect What You Need via Reasoning** — API
- `openalex:W4389523832`: **Evaluating Object Hallucination in Large Vision-Language Models** — API
- `openalex:W4389977164`: **G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model** — API
- `openalex:W4389983382`: **Silkie: Preference Distillation for Large Visual Language Models** — API
- `openalex:W4401042360`: **LMFlow: An Extensible Toolkit for Finetuning and Inference of Large Foundation M** — API
- `openalex:W4402670859`: **Aligning Large Multimodal Models with Factually Augmented RLHF** — API
- `openalex:W4402671571`: **Arithmetic Control of LLMs for Diverse User Preferences: Directional Preference ** — API
- `openalex:W4402727405`: **RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Corre** — API
- `openalex:W4402753774`: **Mitigating Object Hallucinations in Large Vision-Language Models through Visual ** — API
- `openalex:W4402781286`: **PerceptionGPT: Effectively Fusing Visual Perception Into LLM** — API
- `openalex:W4404575065`: **ShareGPT4V: Improving Large Multi-modal Models with Better Captions** — API
- `openalex:W4404782342`: **The Instinctive Bias: Spurious Images lead to Illusion in MLLMs** — API
- `openalex:W4404783010`: **MLLM-Protector: Ensuring MLLM’s Safety without Hurting Performance** — API
- `openalex:W1522301498`: **Adam: A Method for Stochastic Optimization** — API
- `openalex:W1533861849`: **Understanding the difficulty of training deep feedforward neural networks** — API
- `openalex:W1544827683`: **Teaching Machines to Read and Comprehend** — API
- `openalex:W1843891098`: **A Neural Attention Model for Abstractive Sentence Summarization** — API
- `openalex:W1973435495`: **Learning to rank for information retrieval** — API
- `openalex:W2012318340`: **Optimum polynomial retrieval functions based on the probability ranking principl** — API
- `openalex:W2020237802`: **Automatic Combination of Multiple Ranked Retrieval Systems** — API
- `openalex:W2047221353`: **Optimizing search engines using clickthrough data** — API
- `openalex:W2081265723`: **Hedge Trimmer** — API
- `openalex:W2108325777`: **Automatic evaluation of machine translation quality using longest common subsequ** — API
- `openalex:W2152314154`: **Accurately interpreting clickthrough data as implicit feedback** — API
- `openalex:W2170973209`: **Semi-supervised Sequence Learning** — API
- `openalex:W2176263492`: **Sequence Level Training with Recurrent Neural Networks** — API
- `openalex:W2467173223`: **Abstractive Sentence Summarization with Attentive Recurrent Neural Networks** — API
- `openalex:W2487501366`: **An Actor-Critic Algorithm for Sequence Prediction** — API
- `openalex:W2512971201`: **Deep Neural Networks for YouTube Recommendations** — API
- `openalex:W2525778437`: **Google's Neural Machine Translation System: Bridging the Gap between Human and M** — API