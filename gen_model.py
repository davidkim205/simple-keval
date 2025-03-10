import argparse
import json
import os

from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="model name")
    parser.add_argument('--data', type=str, default='data.jsonl')
    parser.add_argument('--num_samples', default=10000, type=int, help='num samples')
    parser.add_argument('--output', type=str, default='results/')

    args = parser.parse_args()
    print(args)

    output_path = args.output

    data_path = args.data.split('/')[-1].split('.jsonl')[0]
    model_name = args.model.split('/')[-2] + '__' + args.model.split('/')[-1]
    output_file_path = os.path.join(output_path, f"{data_path}__{model_name}.jsonl")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    llm = LLM(model=args.model, max_model_len=4096)
    llm.reset_prefix_cache()
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.8)

    data_list = read_jsonl(args.data)

    system_prompt=""
    is_first=True
    for index, item in tqdm(enumerate(data_list)):
        if index >= args.num_samples:
            break

        for pair in item["pairs"]:
            prompt =f"{system_prompt}\n질문:{pair['prompt']} "

            conversation = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            outputs = llm.chat(messages=conversation, sampling_params=sampling_params)
            answer = outputs[0].outputs[0].text
            pair["gen"] = answer

        if is_first:
            mode='w'
            is_first=False
        else:
            mode='a'
        write_jsonl(output_file_path, item, mode)




if __name__ == '__main__':
    main()
