import argparse
import json

from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="model name")
    parser.add_argument('--repo', type=str, default='davidkim205/ko-bench')
    parser.add_argument('--data', type=str, default='pairs_ko_question.jsonl')
    parser.add_argument('--num_samples', default=10000, type=int, help='num samples')
    parser.add_argument('--output', type=str, default='results/')

    args = parser.parse_args()
    print(args)

    data_list = load_dataset(args.repo, data_files=args.data)['train'].to_list()
    print(f'Load {args.repo}/{args.data}', '\n')

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # {output_dir}/{modelname}__{testset}__result.jsonl
    output_file_path = output_path / (args.model.split('/')[-1] + '__' + args.data.split('/')[-1].replace('.jsonl', '') + '__' + 'result.jsonl')
    print('Output Path:', output_file_path, '\n')

    llm = LLM(model=args.model, max_model_len=4096)
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.8)

    f = open(output_file_path, mode='w')
    for index, item in tqdm(enumerate(data_list), total=len(data_list)):
        if index >= args.num_samples:
            break
        item['pairs'] = item['pairs'][:1] # only single-turn

        for pair in item["pairs"]: 
            conversation = [
                {
                    "role": "user",
                    "content": pair['prompt']
                }
            ]
            outputs = llm.chat(messages=conversation, sampling_params=sampling_params)
            answer = outputs[0].outputs[0].text
            pair["gen"] = answer

        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()
    f.close()


if __name__ == '__main__':
    main()
