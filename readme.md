# Simple-Keval

The simple-keval project facilitates the evaluation of responses generated by LLMs to Ko-Bench questions using the LLM-as-a-judge approach.

Ko-Bench is a localized set of questions derived from MT-bench, aimed at providing a refined assessment of Korean LLMs. However, this project is limited to single-turn interactions. 

The evaluation leverages the keval and kgrammar models, focusing on two aspects: keval examines whether the response is appropriate and accurate, while kgrammar identifies any grammatical errors in the response. Both models work to quantify and measure the quality of the responses.

| [**Model**](https://huggingface.co/collections/davidkim205/keval-2-67ac5400f5eef4984cc5dbbb) | [**Paper**](https://davidkim205.github.io/keval.html) | [**Code**](https://github.com/davidkim205/simple-keval) |

| [**Space(keval)**](https://huggingface.co/spaces/davidkim205/keval-2) | [**Space(kgrammar)**](https://huggingface.co/spaces/davidkim205/kgrammar-2) | [**Dataset(keval)**](https://huggingface.co/datasets/davidkim205/keval-testset) | [**Dataset(kgrammar)**](https://huggingface.co/datasets/davidkim205/kgrammar-testset) |

## Contents

- [Installation](#installation)
- [Keval](#keval)
- [Datasets](#datasets)
- [Models](#models)
- [Citation](#citation)

## Installation

We are using conda virtual environments, but it is acceptable to use other virtual environments that integrate with Python.

To set up the environment, follow these steps:

```
conda create -n simple-keval python=3.12
conda activate simple-keval
pip install -r requirements.txt
```

## Keval

### Step 1. Generate model answers

Generate the answers for the Ko-Bench testset.

```
python gen_model.py [MODEL] --data [DATA] --num_samples [NUM_SAMPLES] --output [OUTPUT]
```

- **Parameters**:
  - `[MODEL]`: Path to the model. Local folder or Hugging Face repo ID.
  - `[DATA]`: Path to the testset. Local folder or Hugging Face repo ID.
  - `[NUM_SAMPLES]`: Number of samples to extract (ordered).
  - `[OUTPUT]`: Directory for results.

e.g.,

```
python gen_model.py google/gemma-2-9b-it --data davidkim205/ko-bench --num_samples 10000 --output results/
```

The answers are saved in `results/google__gemma-2-9b-it__result.jsonl`

### Step 2. Generate kgrammar judgements

The kgrammar detects Korean grammar errors in responses and quantifies them for evaluation, using only the assistant's answer for judgment.
The scoring criteria are as follows: if there are no grammatical errors, you score 1 point; otherwise, you score 0 points.

```
python eval_kgrammar.py --data [DATA] --model [MODEL] --output [OUTPUT]
```

- **Parameters**:
  - `[DATA]`: Path to the file to be evaluated.
  - `[MODEL]`: Path to the kgrammar model. Local folder or Hugging Face repo ID.
  - `[OUTPUT]`: Directory for kgrammar results.

e.g.,

```
python eval_kgrammar.py --data results/google__gemma-2-9b-it__result.jsonl --model davidkim205/kgrammar-2-1b --output results_kgrammar/
```

The answers are saved in `results_kgrammar/google__gemma-2-9b-it__kgrammar.jsonl`

### Step 3. Generate keval judgements

The keval judges the relevance, accuracy, and usefulness of responses, assigning a score from 0 to 10. It makes this judgment in context, considering the question, the reference answer, and the assistant's answer.

```
python eval_keval.py --data [DATA] --model [MODEL] --output [OUTPUT]
```

- **Parameters**:
  - `[DATA]`: Path to the file to be evaluated.
  - `[MODEL]`: Path to the keval model. Local folder or Hugging Face repo ID.
  - `[OUTPUT]`: Directory for keval results.

e.g.,

```
python eval_keval.py --data results/google__gemma-2-9b-it__result.jsonl --model davidkim205/keval-2-1b --output results_keval/
```

The answers are saved in `results_keval/google__gemma-2-9b-it__keval.jsonl`

### Step 4. Show Ko-Bench scores

It is possible to check the performance metrics of the assistant's answers for each testset used.

e.g.,

```
python score.py --keval results_keval/ --kgrammar results_kgrammar/
```

Example scoring output:

```
# Testset: pairs_ko_question 

|    | model               |   average | keval     | kgrammar   |
|---:|:--------------------|----------:|:----------|:-----------|
|  1 | gemma-2-9b-it       |      0.75 | 0.66 (80) | 0.84 (80)  |
|  0 | ko-gemma-2-9b-it-v2 |      0.74 | 0.63 (80) | 0.85 (80)  |
```

## Datasets

- [Ko-Bench Dataset](https://huggingface.co/datasets/davidkim205/ko-bench)
- [keval-test Dataset](https://huggingface.co/datasets/davidkim205/keval-testset)
- [kgrmmar-test Dataset](https://huggingface.co/datasets/davidkim205/kgrammar-testset)

## Models

We provide an open model for judgment, and access to the 9b model can be requested if needed.

- [keval-2-1b](https://huggingface.co/davidkim205/keval-2-1b)
- [keval-2-3b](https://huggingface.co/davidkim205/keval-2-3b)
- [keval-2-9b](https://huggingface.co/davidkim205/keval-2-9b) *(requires approval)*
- [kgrammar-2-1b](https://huggingface.co/davidkim205/kgrmmar-2-1b)
- [kgrammar-2-3b](https://huggingface.co/davidkim205/kgrmmar-2-3b)
- [kgrammar-2-9b](https://huggingface.co/davidkim205/kgrmmar-2-9b) *(requires approval)*

## Citation

If our project has been helpful for evaluating LLMs, please consider citing the [technical report](https://davidkim205.github.io/keval.html) available on our blog.

```
@misc{kim2025keval,
  title={keval and kgrammar: Offline-Ready Evaluation Frameworks for Korean AI Models},
  author={Changyeon Kim and Yehee Lim and Bumsu Jung and Yeonsu Ho},
  year={2025},
  howpublished={Online; accessed via a company blog},
  url={https://davidkim205.github.io/keval.html}
  note={2Digit AI Research Blog}
}
```
