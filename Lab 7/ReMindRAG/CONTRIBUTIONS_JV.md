# Individual Reflection - Lab 7
**Author: Joel Vinas**

This reflection documents my individual contributions and learning outcomes during the reproduction and audit of the ReMindRAG repository.

## Specific Technical Contributions
My primary technical contribution was an **independent attempt to implement and replicate the ReMindRAG repository** from scratch. During this process, I focused on:
- Setting up the Python 3.13 environment and verifying the pinned dependencies in `requirements.txt`.
- Implementing a configuration-driven execution model by pivoting from the suggested `api_key.json` to a more robust and secure `.env` approach.
- Provided critical feedback during our team regroup session, where I detailed the specific hurdles encountered during the extraction phase and suggested ways to overcome them using better logging and environment management.

## Challenges Encountered
The reproduction effort was met with several significant challenges, which highlighted the complexities of modern AI-based systems:
- **API Key Configuration**: The original documentation’s reliance on `api_key.json` was prone to formatting errors. Navigating this and implementing a `.env` loader was a necessary hurdle.
- **API Rate Limits**: The most critical failure occurred during the knowledge graph extraction phase. The high volume of parallel LLM calls frequently triggered `429 Too Many Requests` errors, requiring a careful balance of chunk sizes and request timing.
- **GPU Acceleration on Windows**: Ensuring that the Nomic embedding model utilized the GPU (via CUDA) required manual verification of PyTorch build versions, as the default installations often defaulted to CPU-only performance.

## What I Learned About Reproducibility
Reproducing this Git repository underscored that **reproducibility is not just about the code, but about the environment and the entry points.** Successfully replicating the system involved:
- Validating that pinned dependencies are non-negotiable for cross-platform stability.
- Recognizing that a lack of automated setup scripts (like the `reproduce.ps1` I eventually created) significantly increases the barrier to entry for external collaborators.

## How Agentic AI Tools Influenced My Workflow
The use of **Antigravity** was crucial to the success of this implementation. As an agentic tool, it allowed me to:
- **Automate the Audit**: Rapidly scan the repository for missing files and directories.
- **Bridge Documentation Gaps**: Generate helper scripts (`reproduce.ps1`) and missing documentation (`RUN.md`, `REPRO_AUDIT.md`) directly within the context of the codebase.
- **Iterate Rapidly**: The ability to perform multi-file edits and verification steps allowed me to focus on high-level architectural fixes rather than manual boilerplate.

## Strengthening Understanding through Reproduction
This was one of my **first attempts to replicate a complex AI model repository from GitHub**, and it significantly strengthened my understanding of Retrieval-Augmented Generation (RAG). By manually stepping through the KG traversal logic and extraction pipelines, I moved beyond theoretical knowledge to a practical understanding of how LLMs and Graph databases interact. This experience has built my confidence in evaluating and adopting modular AI architectures for future projects.
