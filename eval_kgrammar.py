from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import json
import os
import re


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


def get_wrong_count(text):
    match = re.search(r'<wrong count>\s*(\d+)\s*</wrong count>', text)
    try:
        return int(match.group(1))
    except:
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='results/gen_model__pairs_nuclear-education-think-qa_test__gemma-2-9b-it.jsonl')
    parser.add_argument('--output', type=str, default='results_kgrammar/')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    output_file_path = args.output + args.data.split('/')[-1].replace("gen_model", "kgrammar")
    if os.path.dirname(args.data).endswith('norag'):
        output_file_path = os.path.splitext(output_file_path)[0] + '__norag.jsonl'

    llm = LLM(model='davidkim205/kgrammar-2-9b', max_model_len=4096)
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.8)

    input_file_path = args.data
    data_list = read_jsonl(input_file_path)

    judge_grammar = "한국어 문맥상 부자연스러운 부분을 찾으시오. 오류 문장과 개수는 <incorrect grammar> </incorrect grammar> tag, 즉 <incorrect grammar> - 오류 문장과 설명 </incorrect grammar> 안에 담겨 있으며, <wrong count> </wrong count> tag, 즉 <wrong count> 오류 개수 </wrong count> 이다."
    is_first=True
    for item in tqdm(data_list):
        for pair in item["pairs"]:
            conversation = [
                {
                    "role": "user",
                    "content": f"{judge_grammar}\n{pair['gen']}"
                }
            ]

            outputs = llm.chat(messages=conversation, sampling_params=sampling_params)
            answer = outputs[0].outputs[0].text
            pair["kgrammar"] = answer

        if is_first:
            mode='w'
            is_first=False
        else:
            mode='a'
        write_jsonl(output_file_path, item, mode)




if __name__ == '__main__':
    main()
