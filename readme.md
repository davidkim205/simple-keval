# Simple-keval

This repository contains scripts and tools for evaluating Korean language models using the `keval` and `kgrammar` models. The evaluation is based on the keval-2 models, which leverages a Large Language Model-as-a-judge approach for accurate assessment.


| [**Model**](https://huggingface.co/collections/davidkim205/keval-2-67ac5400f5eef4984cc5dbbb) | [**Paper**](https://davidkim205.github.io/keval.html) | [**Code**](https://github.com/davidkim205/simple-keval) |

| [**Space(keval)**](https://huggingface.co/spaces/davidkim205/keval-2) | [**Space(kgrammar)**](https://huggingface.co/spaces/davidkim205/kgrammar-2) | [**Dataset(keval)**](https://huggingface.co/datasets/davidkim205/keval-testset) | [**Dataset(kgrammar)**](https://huggingface.co/datasets/davidkim205/kgrammar-testset) |


## Datasets

- [Ko-Bench Dataset](https://huggingface.co/datasets/davidkim205/ko-bench)
- [keval-test Dataset](https://huggingface.co/datasets/davidkim205/keval-testset)
- [kgrmmar-test Dataset](https://huggingface.co/datasets/davidkim205/kgrammar-testset)

## Models

An advanced evaluation model specifically designed for assessing Korean language models.
Utilizes the Gemma2-9b architecture with enhancements through Supervised Fine-Tuning (SFT) and Direct Policy Optimization (DPO).
Trained on the Ko-bench dataset, inspired by MT-bench, tailored for Korean linguistic nuances.

- [keval-2-1b](https://huggingface.co/davidkim205/keval-2-1b)
- [keval-2-3b](https://huggingface.co/davidkim205/keval-2-3b)
- [keval-2-9b](https://huggingface.co/davidkim205/keval-2-9b) *(requires approval)*
- [kgrammar-2-1b](https://huggingface.co/davidkim205/kgrmmar-2-1b)
- [kgrammar-2-3b](https://huggingface.co/davidkim205/kgrmmar-2-3b)
- [kgrammar-2-9b](https://huggingface.co/davidkim205/kgrmmar-2-9b) *(requires approval)*

## How to Use

1. **Installation**

   Run the following command to install the required dependencies:

    ```bash
    conda create -n simple-keval python=3.12
    conda activate simple-keval
    pip install -r requirements.txt
    ```

2. **Generate Model Outputs**

   Use `gen_model.py` to generate initial model outputs. Modify arguments as needed:

   ```bash
   python gen_model.py <model_path>  --data <DATA>
   ```
   if you want to evaluate with ko-bench, do the following:
   ```bash
   python gen_model.py <model_path> 
   ```

3. **Evaluate with Keval**

   Use `eval_keval.py` to evaluate the model's response quality:

   ```bash
   python eval_keval.py --data results/<generated_output_file>
   ```

4. **Evaluate with Kgrammar**

   Use `eval_kgrammar.py` to assess grammatical accuracy:

   ```bash
   python eval_kgrammar.py --data results/<generated_output_file>
   ```

5. **Aggregate Scores**

   Use `score.py` to aggregate and generate final evaluation scores:

   ```bash
   python score.py --keval results_keval/ --kgrammar results_kgrammar/
   ```

```
| model                                             |   average | keval     | kgrammar   |
|:--------------------------------------------------|----------:|:----------|:-----------|
| davidkim205__ko-gemma-2-9b-it-v2                  |      0.75 | 0.60 (80) | 0.90 (78)  |
| rtzr__ko-gemma-2-9b-it                            |      0.74 | 0.59 (80) | 0.89 (80)  |
| NCSOFT__Llama-VARCO-8B-Instruct                   |      0.74 | 0.62 (80) | 0.86 (79)  |
| LGAI-EXAONE__EXAONE-3.5-7.8B-Instruct             |      0.73 | 0.58 (80) | 0.87 (79)  |
| google__gemma-2-9b-it                             |      0.72 | 0.62 (79) | 0.81 (78)  |
| MLP-KTLim__llama-3-Korean-Bllossom-8B             |      0.70 | 0.51 (80) | 0.88 (78)  |
| KAERI-MLP__llama-3.1-Korean-AtomicGPT-Bllossom-8B |      0.70 | 0.55 (80) | 0.84 (77)  |
| Qwen__Qwen2-7B-Instruct                           |      0.69 | 0.48 (80) | 0.89 (80)  |
| dnotitia__Llama-DNA-1.0-8B-Instruct               |      0.57 | 0.42 (80) | 0.73 (77)  |
```

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
