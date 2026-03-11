# Individual Reflection - Lab 7
**Author: Joel Vinas**
**GitHub Repo: https://github.com/joelvinas/COMP-SCI_5542/tree/fc47674174ae112a1eb525fee8e55a534aefb241/Lab%207**
**Team Report: https://github.com/joelvinas/COMP-SCI_5542/blob/7b37b587af613cd4b19a884d56c650c80415e9eb/Lab%207/ReMindRAG/reports/TEAM_REPORT.md**

This reflection documents my individual contributions and learning outcomes during the reproduction, audit, and team-based enhancement of the ReMindRAG repository.

## Specific Technical Contributions
In addition to my independent attempt to replicate the base repository, I took primary ownership of **LLM Reasoning and System Robustness** in our team implementation. My key technical contributions include:

- **Reasoning Injection**: Designed the upgrade path for implementing **Chain-of-Thought (CoT)** and **Self-Consistency** methodologies into the entity extraction agent. This significantly improved the quality of the knowledge graph by ensuring the LLM "thinks" through relationships before extracting them.
- **Cost Control**: Refactored the evaluation orchestrator (`eval_LooGLE.py`) to allow for **dynamic switching of judge models**. This enabled us to override hardcoded `gpt-4o` defaults with `gpt-4o-mini`, drastically reducing the cost of running large evaluation batches without sacrificing result quality.
- **System Improvements**: Integrated a global `--seed` parameter to enforce **strict deterministic randomness** across all PyTorch and NumPy operations. This was critical for ensuring that evaluation results were consistent across different runs and environments.
- **Team Feedback**: Provided detailed feedback during regroup sessions based on my initial hurdles, facilitating the shift toward environment-variable-based configuration and robust logging.

## Challenges Encountered
The reproduction and enhancement effort was met with several significant challenges:
- **API Key Configuration**: Documented the shift from `api_key.json` to `.env` for better security and portability.
- **API Rate Limits**: Navigated the frequent `429 Too Many Requests` errors during the token-intensive Information Extraction phase by optimizing chunk sizes.
- **GPU Acceleration**: Verified the necessity of CUDA-enabled PyTorch for localized embedding models, ensuring team-wide performance parity on Windows.

## What I Learned About Reproducibility
Reproducing this repository taught me that true reproducibility requires **automation and strict versioning**. Accomplishing the replication of the KG traversal logic proved that:
- Deterministic seeding (via the `--seed` parameter I implemented) is vital for scientific verification in AI.
- Automated entry points (like `reproduce.ps1`) are essential for lowering the friction of collaborative research.

## How Agentic AI Tools Influenced My Workflow
The use of **Antigravity** was crucial to this project. It served as a force multiplier by:
- **Auditing and File Generation**: Quickly identifying missing documentation and generating artifacts like `RUN.md` and `REPRO_AUDIT.md`.
- **Code Refactoring**: assisting in the refactoring of `eval_LooGLE.py` and the integration of seeding logic across multiple modules.
- **Workflow Orchestration**: Allowing me to focus on high-level reasoning and cost-control strategies while the agent handled the boilerplate of repository structure.

## Strengthening Understanding through Reproduction
As this was one of my first attempts to replicate a complex AI model repository from GitHub, the experience was transformative. Moving beyond a "black box" view of RAG into the specifics of LLM-guided graph traversal has strengthened my ability to build robust, efficient, and reproducible AI systems.
