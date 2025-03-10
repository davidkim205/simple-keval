import argparse
import json
import os
import re

from vllm import LLM, SamplingParams
from tqdm import tqdm


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
    parser.add_argument('--data', type=str, default='results/ko-gemma-2-9b-it-v2__pairs_ko_question__result.jsonl')
    parser.add_argument('--output', type=str, default='results_kgrammar/')
    args = parser.parse_args()
    print(args)

    data_list = read_jsonl(args.data)

    os.makedirs(args.output, exist_ok=True)

    output_file_path = args.output + args.data.split('/')[-1].replace("__result.jsonl", "__kgrammar.jsonl")

    llm = LLM(model='davidkim205/kgrammar-2-9b', max_model_len=4096)
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.8)

    judge_grammar = "한국어 문맥상 부자연스러운 부분을 찾으시오. 오류 문장과 개수는 <incorrect grammar> </incorrect grammar> tag, 즉 <incorrect grammar> - 오류 문장과 설명 </incorrect grammar> 안에 담겨 있으며, <wrong count> </wrong count> tag, 즉 <wrong count> 오류 개수 </wrong count> 이다."
    
    f = open(output_file_path, mode='w')
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

        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()
    f.close()




if __name__ == '__main__':
    main()
