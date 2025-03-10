from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import json
import ast
import os
import re


one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")

# Define a function to read from jsonl file
def read_jsonl(file_path):
    data_list=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list

# Define a function to write to jsonl file
def write_jsonl(file_path, data, mode='a'):
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def get_score(judgment):
    match = re.search(one_score_pattern, judgment)
    if not match:
        match = re.search(one_score_pattern_backup, judgment)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1
    return rating


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='results/gen_model__pairs_nuclear-education-think-qa_test__gemma-2-9b-it.jsonl')
    parser.add_argument('--output', type=str, default='results_keval/')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    output_file_path = args.output + args.data.split('/')[-1].replace("gen_model", "kgrammar")
    if os.path.dirname(args.data).endswith('norag'):
        output_file_path = os.path.splitext(output_file_path)[0] + '__norag.jsonl'

    llm = LLM(model='davidkim205/keval-2-9b', max_model_len=4096)
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.8)

    input_file_path = args.data
    data_list = read_jsonl(input_file_path)


    is_first=True
    for item in tqdm(data_list):
        for pair in item["pairs"]:
            question = pair['prompt']
            ref_answer_1 = pair['output']
            answer = pair['gen']

            prompt = f"[지시]\n공정한 심판으로서 아래에 표시된 사용자 질문에 대한 AI 어시스턴트의 응답 품질을 평가해주세요. 질문과 대답의 언어가 동일하지 않으면 무조건 0점입니다. 평가는 정확성과 유용성을 고려해야 합니다. 참고 답변과 어시스턴트의 답변이 제공될 것입니다. 평가를 시작하기 위해 어시스턴트의 답변을 참고 답변과 비교하세요. 각 답변의 실수를 식별하고 수정하세요. 가능한 한 객관적으로 평가하세요. 설명을 제공한 후 다음 형식을 엄격히 따라 응답을 1점에서 10점 사이로 평가해야 합니다: \"[[rating]]\", 예를 들어: \"Rating: [[5]]\".\n\n[질문]\n{question}\n\n[참조 답변의 시작]\n{ref_answer_1}\n[참조 답변의 끝]\n\n[어시스턴트 답변의 시작]\n{answer}\n[어시스턴트 답변의 끝]"

            conversation = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            outputs = llm.chat(messages=conversation, sampling_params=sampling_params)
            answer = outputs[0].outputs[0].text
            pair["keval"] = answer

        if is_first:
            mode='w'
            is_first=False
        else:
            mode='a'
        write_jsonl(output_file_path, item, mode)




if __name__ == '__main__':
    main()
