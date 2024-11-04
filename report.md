# TinyStarCoder Code Completion Project Report

## Overview
This project aims to explore code completion using the TinyStarCoder model. The goal was to generate and evaluate code completions for given prefixes and suffixes.

## Methodology
1. **Dataset Generation**: 
   - Snippets were collected from personal projects.
   - Each snippet was split into prefix, middle (missing), and suffix parts.

2. **Model Inference**:
   - The TinyStarCoder model was used to generate completions for the middle part based on the prefix and suffix.

3. **Evaluation**:
   - Completions were evaluated using exact match and BLEU scores.

## Findings
- The model achieved high exact match rates (100% in test cases).
- BLEU scores indicated strong similarity between generated and actual code.

## Conclusion
The project demonstrated the effectiveness of the TinyStarCoder model in generating accurate code completions. Future work could explore other models and larger datasets for improved robustness.
