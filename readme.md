# simple-keval


## installation
```  
conda create -n simple-keval python=3.12
conda activate simple-keval
pip install -r requirements.txt
```

## usage
모델 평가 파이프라인은 아래의 4단계로 진행한다.

1. llm judge 질문 데이터셋에 대한 답변 생성
2. kgrammar 평가
3. keval 평가
4. score 확인

### 1. gen_model
llm judge 테스트셋의 답변을 생성하며, 여러 task로 분류되어 있다.
테스트 데이터셋은 아래에 공개되어 있다.
https://huggingface.co/datasets/davidkim205/ko-bench

```
python gen_model.py [model] [--repo davidkim205/ko-bench] [--data pairs_ko_question.jsonl] [--num_samples 10000] [--output results/]
```

인자는 아래와 같다.

- model: 모델 경로
- repo: 공개 테스트셋 저장소
- data: 데이터셋 이름
- num_samples: 샘플링 개수(순서대로 추출)
- output: 결과 디렉토리

결과 스키마 예시

```json
{
  "question_id": 115,
  "category": "math",
  "pairs": [
    {
      "prompt": "터미널에서 버스를 타는 사람들도 있었습니다. 첫 번째 정류장에서는 절반이 내리고 4명이 더 탔습니다. 그리고 두 번째 정류장에서는 6명이 내리고 8명이 더 탔습니다. 세 번째 정류장까지 총 25명이 타고 있었다면, 터미널에서 버스에 몇 명이 탔나요?",
      "refer": "38명",
      "gen": "1. **세 번째 정류장에서의 승객 수를 알려주었습니다:** 세 번째 정류장까지 총 25명이 타고 있었다는 것을 알고 있습니다.\n2. **두 번째 정류장에서의 변화를 고려해야 합니다:** 두 번째 정류장에서 6명이 내리고 8명이 탔으므로, net으로는 2명이 더 탑승했습니다.\n3. **두 번째 정류장 이전 승객 수를 계산합니다:** 세 번째 정류장에서의 승객 수(25명)에서 두 번째 정류장에서 탑승한 2명을 빼줍니다. 즉, 두 번째 정류장 전에 23명이 타고 있었습니다.\n4. **첫 번째 정류장에서의 변화를 고려합니다:** 첫 번째 정류장에서 절반이 내리고 4명이 탔으므로,  net으로는 4명이 탑승한 것보다 적게 탑승했을 것입니다.\n5. **터미널에서의 승객 수를 계산합니다:** 첫 번째 정류장 이전의 승객 수를 알기 위해, 두 번째 정류장 전 승객 수(23명)에서 탑승한 4명을 빼줍니다. 즉, 터미널에서 버스에 탑승한 승객 수는 19명입니다. \n\n\n\n따라서 터미널에서 버스에 **19명**이 탔습니다."
    }
  ]
}
```

### 2. kgrammar judge
kgrammar는 문법적인 측면에서 한국어 답변의 문맥이 자연스러운지 평가하고 오류를 탐지한다.
탐지한 오류의 개수를 정량적인 평가지표로 사용한다.
kgrammar judge 모델을 정량평가한 테스트 데이터셋은 아래에 공개되어 있다.
https://huggingface.co/datasets/davidkim205/kgrammar-testset

```
python eval_kgrammar.py [--data results/ko-gemma-2-9b-it-v2__pairs_ko_question__result.jsonl] [--model davidkim205/kgrammar-2-9b] [--output results_kgrammar/]
```

인자는 아래와 같다.

- data: 평가할 결과파일 경로
- model: kgrammar judge 모델
- output: 결과 디렉토리

결과 스키마 예시

```json
{
  "question_id": 115,
  "category": "math",
  "pairs": [
    {
      "prompt": "터미널에서 버스를 타는 사람들도 있었습니다. 첫 번째 정류장에서는 절반이 내리고 4명이 더 탔습니다. 그리고 두 번째 정류장에서는 6명이 내리고 8명이 더 탔습니다. 세 번째 정류장까지 총 25명이 타고 있었다면, 터미널에서 버스에 몇 명이 탔나요?",
      "refer": "38명",
      "gen": "1. **세 번째 정류장에서의 승객 수를 알려주었습니다:** 세 번째 정류장까지 총 25명이 타고 있었다는 것을 알고 있습니다.\n2. **두 번째 정류장에서의 변화를 고려해야 합니다:** 두 번째 정류장에서 6명이 내리고 8명이 탔으므로, net으로는 2명이 더 탑승했습니다.\n3. **두 번째 정류장 이전 승객 수를 계산합니다:** 세 번째 정류장에서의 승객 수(25명)에서 두 번째 정류장에서 탑승한 2명을 빼줍니다. 즉, 두 번째 정류장 전에 23명이 타고 있었습니다.\n4. **첫 번째 정류장에서의 변화를 고려합니다:** 첫 번째 정류장에서 절반이 내리고 4명이 탔으므로,  net으로는 4명이 탑승한 것보다 적게 탑승했을 것입니다.\n5. **터미널에서의 승객 수를 계산합니다:** 첫 번째 정류장 이전의 승객 수를 알기 위해, 두 번째 정류장 전 승객 수(23명)에서 탑승한 4명을 빼줍니다. 즉, 터미널에서 버스에 탑승한 승객 수는 19명입니다. \n\n\n\n따라서 터미널에서 버스에 **19명**이 탔습니다.",
      "kgrammar": "<incorrect grammar>```\n- \"net으로는\"은 한국어 문맥에서 부자연스러운 영어 사용입니다. \"net\"이라는 단어는 한국어로 \"네트\"로 대체되어야 합니다.\n</incorrect grammar> <wrong count>2</wrong count>\n```\n"
    }
  ]
}
```

### 3. keval judge

keval은 한국어 답변의 정성적인 품질을 분석하고 정량적으로 점수를 측정하여 평가지표로 사용한다.
keval judge 모델을 정량평가한 테스트 데이터셋은 아래에 공개되어 있다.
https://huggingface.co/datasets/davidkim205/keval-testset

```
python gen_model.py [--data results/ko-gemma-2-9b-it-v2__pairs_ko_question__result.jsonl] [--model davidkim205/keval-2-9b] [--output results_keval/]
```

인자는 아래와 같다.

- data: 평가할 결과파일 경로
- model: keval judge 모델
- output: 결과 디렉토리

결과 스키마 예시

```json
{
  "question_id": 115,
  "category": "math",
  "pairs": [
    {
      "prompt": "터미널에서 버스를 타는 사람들도 있었습니다. 첫 번째 정류장에서는 절반이 내리고 4명이 더 탔습니다. 그리고 두 번째 정류장에서는 6명이 내리고 8명이 더 탔습니다. 세 번째 정류장까지 총 25명이 타고 있었다면, 터미널에서 버스에 몇 명이 탔나요?",
      "refer": "38명",
      "gen": "1. **세 번째 정류장에서의 승객 수를 알려주었습니다:** 세 번째 정류장까지 총 25명이 타고 있었다는 것을 알고 있습니다.\n2. **두 번째 정류장에서의 변화를 고려해야 합니다:** 두 번째 정류장에서 6명이 내리고 8명이 탔으므로, net으로는 2명이 더 탑승했습니다.\n3. **두 번째 정류장 이전 승객 수를 계산합니다:** 세 번째 정류장에서의 승객 수(25명)에서 두 번째 정류장에서 탑승한 2명을 빼줍니다. 즉, 두 번째 정류장 전에 23명이 타고 있었습니다.\n4. **첫 번째 정류장에서의 변화를 고려합니다:** 첫 번째 정류장에서 절반이 내리고 4명이 탔으므로,  net으로는 4명이 탑승한 것보다 적게 탑승했을 것입니다.\n5. **터미널에서의 승객 수를 계산합니다:** 첫 번째 정류장 이전의 승객 수를 알기 위해, 두 번째 정류장 전 승객 수(23명)에서 탑승한 4명을 빼줍니다. 즉, 터미널에서 버스에 탑승한 승객 수는 19명입니다. \n\n\n\n따라서 터미널에서 버스에 **19명**이 탔습니다.",
      "keval": "어시스턴트의 답변은 질문에 대한 정확한 해결책을 제공하지 못했습니다. 문제는 터미널에서 시작하여 정류장을 거쳐 세 번째 정류장까지의 승객 수 변화를 추적하는 것입니다. 어시스턴트는 잘못된 계산을 통해 터미널에서 19명이 탔다고 잘못된 결론을 내렸습니다.\n\n참조 답변은 정확하게 문제를 해결하였고, 터미널에서 38명이 탔다는 것을 올바르게 계산했습니다.\n\n어시스턴트의 답변에서의 실수:\n1. 두 번째 정류장에서 6명이 내리고 8명이 탔다는 정보를 잘못 해석하여, 두 번째 정류장에서 2명이 더 탑승했다고 잘못 계산했습니다.\n2. 첫 번째 정류장에서 절반이 내리고 4명이 탔다는 정보를 제대로 활용하지 못했습니다.\n\n따라서 어시스턴트의 답변은 정확성과 유용성 모두에서 부족합니다.\n\nRating: [[1]]\n"
    }
  ]
}
```
    parser.add_argument('--keval', type=str, default='results_keval/')
    parser.add_argument('--kgrammar', type=str, default='results_kgrammar/')
### 4. aggregate
평가한 모델의 keval, kgrammar 점수를 확인한다.

```
python score.py [--keval results_keval/] [--kgrammar results_kgrammar]
```

결과 예시
```
# Testset: pairs_ko_question 

|    | model               |   average | keval     | kgrammar   |
|---:|:--------------------|----------:|:----------|:-----------|
|  1 | gemma-2-9b-it       |      0.75 | 0.66 (80) | 0.84 (80)  |
|  0 | ko-gemma-2-9b-it-v2 |      0.74 | 0.63 (80) | 0.85 (80)  |
```