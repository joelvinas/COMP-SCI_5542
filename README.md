# COMP-SCI_5542
Code related to UMKC Course COMP-SCI 5542 - Big Data Analytics &amp; Applications

## Lab 2 — Advanced RAG Results ##
### Results table (Query × Method × Precision@5 / Recall@10) ###
	P@5_keyword	R@10_keyword	P@5_vector	R@10_vector	P@5_hybrid	R@10_hybrid	query	alpha_used	num_relevant_labeled
0	0.0	0.0	0.0	0.0	0.0	0.0	Q1: Which university produced a new method to ...	0.2	0
1	0.0	0.0	0.0	0.0	0.0	0.0	Q2: How long does it take to naturally regrow ...	0.2	0
2	0.0	0.0	0.0	0.0	0.0	0.0	Q3 (ambiguous): How could this technique chang...	0.2	0

### Screenshots: chunking comparison, reranking before/after, prompt-only vs RAG answers ###
<image></image>

### Reflection (3–5 sentences): one failure case, which layer failed, one concrete fix ###
#### Chunking failure ####
I was unable to determine how the code provided chunked the data. Neither the provided documentation nor the lectures from the PowerPoint provided clarity on what was wrong with the code-base. Google Gemini was unable to provide clarity on what was wrong with the chunking within the requisite time.

#### Retrieval failure ####
The code provided was unable to retrieve the proper answer from any of the prompts. Gemini was unable to determine the fault and provide directions on how to repair the faulty code within the requisite time.

#### Re-ranking failure ####
I was unable to determine how the re-ranking was performed, if at all. I feel the provided code is faulty, and Gemini was not able to provide directions on how to repair the fault within the requisite time.

#### Generation failure ####
The code was able successfully generate a human-readable answer. Unfortunately, it was not the correct answer for any of the prompts.
