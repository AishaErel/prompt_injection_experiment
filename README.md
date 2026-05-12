# RAG Prompt Injection Defense System

This project explores security vulnerabilities in Retrieval-Augmented Generation (RAG) systems, specifically **prompt injection attacks**, and implements a multi-layered defense pipeline as a solution suggestion.

Prompt injection occurs when malicious instructions are embedded in retrieved or user-provided content, causing large language models (LLMs) to deviate from intended behavior. This is especially critical in RAG systems where external documents are not fully trusted.


## Overview

This system implements and evaluates a **hybrid defense pipeline** designed to detect and mitigate prompt injection attacks in RAG-based applications.

The pipeline combines:
- Rule-based filtering
- Heuristic risk scoring
- Threshold-based pruning
- Soft reranking of retrieved documents
- Lightweight LLM-based semantic classification (uses Ollama model)

Unlike strict filtering systems, this approach preserves useful context while reducing exposure to malicious instructions.

---

## Features

- Detection of explicit, subtle, and obfuscated prompt injection attacks
- Multi-stage defense pipeline (rule + statistical + semantic)
- Risk-aware reranking instead of hard filtering
- LLM-assisted classification layer
- Evaluation using precision, recall, accuracy, and retention metrics

---

## Architecture

The system follows this pipeline:
1. Query input
2. Document retrieval (RAG system)
3. Defense pipeline:
   - Rule-based filtering
   - Risk scoring
   - Threshold pruning
   - LLM classification
4. Final reranking and filtering
5. Answer generation / evaluation

## Dataset

The dataset is based on the **Stanford Question Answering Dataset (SQuAD)**, augmented with:
- Benign documents
- Synthetic malicious documents:
  - Explicit prompt injection
  - Subtle instruction embedding
  - Obfuscated attack patterns

## Installation

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
pip install -r requirements.txt
