# MaxSimE — SIGIR 2023  


This repository contains the experimental code used in the SIGIR 2023 paper:

### MaxSimE: Explaining Transformer-based Semantic Similarity via Contextualized Best Matching Token Pairs (Brito and Iser, 2023)

The purpose of this repository is primarily reproducibility rather than reusability.  
The code reflects the exact experimental setup used for the SIGIR 2023 publication and enables faithful replication of the results. It is not designed as a general-purpose or production-ready library.

---

## Purpose of the Codebase

This repository was created to support research reproducibility. Accordingly:

- The code mirrors the exact experimental pipeline used in the SIGIR 2023 paper.  
- It prioritizes fidelity to the publication over modularity or extensibility.  
- It is intended to help readers, reviewers, and researchers replicate the results presented in the paper.

If you plan to adapt or extend this work, please be aware that the code may require restructuring.

---

# Contact

For questions about the code, reproducibility, or the SIGIR 2023 experiments, please open an issue or contact the authors.

# Citation

If you use this code, build upon this work, or reproduce the experimental results, please cite the original SIGIR 2023 paper:

```bibtex
@inproceedings{10.1145/3539618.3592017,
  author = {Brito, Eduardo and Iser, Henri},
  title = {MaxSimE: Explaining Transformer-based Semantic Similarity via Contextualized Best Matching Token Pairs},
  year = {2023},
  isbn = {9781450394086},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3539618.3592017},
  doi = {10.1145/3539618.3592017},
  abstract = {Current semantic search approaches rely on black-box language models, such as BERT, which limit their interpretability and transparency. In this work, we propose MaxSimE, an explanation method for language models applied to measure semantic similarity. Our approach is inspired by the explainable-by-design ColBERT architecture and generates explanations by matching contextualized query tokens to the most similar tokens from the retrieved document according to the cosine similarity of their embeddings. Unlike existing post-hoc explanation methods, which may lack fidelity to the model and thus fail to provide trustworthy explanations in critical settings, we demonstrate that MaxSimE can generate faithful explanations under certain conditions and how it improves the interpretability of semantic search results on ranked documents from the LoTTe benchmark, showing its potential for trustworthy information retrieval.},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages = {2154--2158},
  numpages = {5},
  keywords = {ad-hoc explanations, explainable search, neural models, semantic similarity, trustworthy information retrieval},
  location = {Taipei, Taiwan},
  series = {SIGIR '23}
}
