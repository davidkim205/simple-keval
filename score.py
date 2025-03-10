from eval_kgrammar import get_wrong_count
from eval_keval import get_score
from collections import defaultdict

import pandas as pd
import numpy as np
import argparse
import json
import os


# Define a function to read from jsonl file
def read_jsonl(file_path):
    data_list=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list


def group_files_by_testset(directory):
    testset_results = defaultdict(list)
    for file in os.listdir(directory):
        if file.endswith('.jsonl'):
            testset_name = file.split('__', 1)[0]
            testset_results[testset_name].append(os.path.join(directory, file))
    return testset_results


def calculate_scores(files, keval_dir, kgrammar_dir):
    scores = {}

    for file in files:
        dirname, filename = os.path.split(file)
        model_name = filename.split('__')[1] + '/' + filename.split('__')[2].replace('.jsonl', '')
        if model_name not in scores:
            scores[model_name] = {}

        prefix = 'GENQA' if filename.find('_genqa_') != -1 else 'QA' if filename.endswith('norag.jsonl') else 'RAGQA'

        if dirname == keval_dir.rstrip('/'):
            output_key = 'keval'
            score_func = get_score
        elif dirname == kgrammar_dir.rstrip('/'):
            output_key = 'kgrammar'
            score_func = get_wrong_count
        else:
            print('skip invalid directory name', dirname)
            continue

        scores[model_name][prefix + ' ' + output_key] = []

        data = read_jsonl(file)
        for line in data:
            for pair in line['pairs']:
                score = score_func(pair[output_key])
                if score < 0:
                    pass
                elif output_key == 'keval': # 0~10 점수를 0~1 사이로 변경
                    score = score * 0.1
                elif output_key == 'kgrammar':
                    score = 1 if score == 0 else 0 # wrong count 0이면 1점
                scores[model_name][prefix + ' ' + output_key].append(score)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keval', type=str, default='results_keval/')
    parser.add_argument('--kgrammar', type=str, default='results_kgrammar/')
    args = parser.parse_args()
    print(args)

    testset_results = defaultdict(list)

    # group by testset
    for directory in [args.keval, args.kgrammar]:
        for testset_name, files in group_files_by_testset(directory).items():
            testset_results[testset_name].extend(files)
            testset_results[testset_name].sort()
    # print(json.dumps(testset_results, indent=4))

    for testset, files in testset_results.items():
        scores = calculate_scores(files, args.keval, args.kgrammar)

        result = []
        for model, score in scores.items():
            average = {}
            for k, v in score.items():
                average[k] = np.mean(v)
            result.append(average)

        df = pd.DataFrame(result, index=scores.keys())
        rag_df = df.loc[:, df.columns.str.contains('RAG')]
        if rag_df.empty:
            df.insert(0, 'average', df.mean(axis=1))
            print("\n\n# Testset:", testset, '\n')
            print('## GENQA')
            print(df.sort_values(by='average', ascending=False).to_markdown(floatfmt='.2f'))
        else:
            df = df.loc[:, ~df.columns.str.contains('RAG')]
            rag_df.insert(0, 'average', rag_df.mean(axis=1))
            df.insert(0, 'average', df.mean(axis=1))
            print("\n\n# Testset:", testset, '\n')
            print('## NORAG')
            print(df.sort_values(by='average', ascending=False).to_markdown(floatfmt='.2f'))
            print()
            print('## RAG')
            print(rag_df.sort_values(by='average', ascending=False).to_markdown(floatfmt='.2f'))


if __name__ == "__main__":
    main()