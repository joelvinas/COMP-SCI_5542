#Ablation Study#

You must compare **at least**:
- **Chunking A (page-based)** vs **Chunking B (fixed-size)**  
- **Sparse** vs **Dense** vs **Hybrid** vs **Hybrid + Rerank** *(dense/rerank can be optional extensions — but include at least sparse + one fusion variant)*  
- **Text-only RAG** vs **Multimodal RAG** (your context must include evidence items)

**Deliverable:** include a final results table in your README:
Results Table:
- Compare page-chunking vs fixed-size (`CHUNK_SIZE` / `CHUNK_OVERLAP`)  

Page-based
|index|id|P@5|R@10|total\_relevant\_chunks|
|---|---|---|---|---|
|0|Q1|0\.4|0\.6666666666666666|3|
|1|Q2|0\.0|0\.0|2|
|2|Q3|0\.6|0\.29411764705882354|17|
|3|Q4|1\.0|0\.19607843137254902|51|

Fixed-size
|index|id|P@5|R@10|total\_relevant\_chunks|
|---|---|---|---|---|
|0|Q1|0\.4|0\.5|4|
|1|Q2|0\.0|0\.0|3|
|2|Q3|0\.6|0\.12903225806451613|31|
|3|Q4|1\.0|0\.050761421319796954|197|

Assessment:
Using a "Page-based" methodology seems to result in a more cohesive retrieval that looks at the larger document contextually.
The Fixed-size seems to run faster, and might work better for larger datasets. In this case, I only used 7 pdfs and 8 pngs.

`Query × Method × Precision@5 × Recall@10 × Faithfulness`

### Quick ablation ideas
- Vary `TOP_K_TEXT`: 2, 5, 10  
- Vary `ALPHA`: 0.2, 0.5, 0.8  


Submission:
1) Your updated dataset: https://github.com/joelvinas/COMP-SCI_5542/tree/main/Lab%203
2) Google Colab Notebook: https://drive.google.com/file/d/1BwGBEe3TUfECNktX50MaMNNPGmqNZW3n/view?usp=sharing
3) A short write‑up: retrieval metrics + faithfulness discussion + ablation
We've developed a remarkable technology in RAG systems, however as an industry we still really don't truly understand how or why it works.
By modifying the parameters for extraction and retrieval, we're able to better understand what factors drive responses that align with expected results.
These knobs and dials vary for larger vs smaller datasets, as well as for text-based vs. graphically intensive documents. The PDF is a popular format, but creates additional problems for extraction and retrieval.