# GitHub backup manifest

The repository backup includes source code, Markdown documentation, teacher traces, judgments, baseline JSONL evaluations, and the 50-episode/two-seed audit. It excludes the local filing corpus, extracted text, retrieval caches, large Tinker logs, and the 170 MB one-step adapter binary; those exclusions are recorded in `.gitignore` and `README.md`.

Before relying on this repository elsewhere, regenerate the corpus/index locally and download the Qwen model through the normal Hugging Face workflow. No credentials or API keys belong in this repository.
