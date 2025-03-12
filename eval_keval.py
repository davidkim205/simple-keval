import argparse
import json

from vllm import LLM, SamplingParams
from tqdm import tqdm
from pathlib import Path


# Define a function to read from jsonl file
def read_jsonl(file_path):
    data_list=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='results/gemma-2-9b-it__pairs_ko_question__result.jsonl')
    parser.add_argument("--model", type=str, default='davidkim205/keval-2-1b', help="keval model")
    parser.add_argument('--output', type=str, default='results_keval/')
    
    args = parser.parse_args()
    print(args)

    data_list = read_jsonl(args.data)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file_path = output_path / args.data.split('/')[-1].replace("__result.jsonl", "__keval.jsonl")

    llm = LLM(model=args.model, max_model_len=4096)
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.8)

    judge_keval = "[지시]\n공정한 심판으로서 아래에 표시된 사용자 질문에 대한 AI 어시스턴트의 응답 품질을 평가해주세요. 질문과 대답의 언어가 동일하지 않으면 무조건 0점입니다. 평가는 정확성과 유용성을 고려해야 합니다. 참고 답변과 어시스턴트의 답변이 제공될 것입니다. 평가를 시작하기 위해 어시스턴트의 답변을 참고 답변과 비교하세요. 각 답변의 실수를 식별하고 수정하세요. 가능한 한 객관적으로 평가하세요. 설명을 제공한 후 다음 형식을 엄격히 따라 응답을 1점에서 10점 사이로 평가해야 합니다: \"[[rating]]\", 예를 들어: \"Rating: [[5]]\".\n\n[질문]\n{question}\n\n[참조 답변의 시작]\n{refer}\n[참조 답변의 끝]\n\n[어시스턴트 답변의 시작]\n{answer}\n[어시스턴트 답변의 끝]"

    f = open(output_file_path, mode='w')
    for item in tqdm(data_list, total=len(data_list)):
        for pair in item["pairs"]:
            prompt = judge_keval.format(question=pair['prompt'], refer=pair['refer'], answer=pair['gen'])

            conversation = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            outputs = llm.chat(messages=conversation, sampling_params=sampling_params)
            answer = outputs[0].outputs[0].text
            pair["keval"] = answer

        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()
    f.close()


if __name__ == '__main__':
    main()
