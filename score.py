import pandas as pd
import numpy as np
import argparse
import json

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
    parser.add_argument('--keval', type=str, default='results_keval/')
    parser.add_argument('--kgrammar', type=str, default='results_kgrammar/')
    args = parser.parse_args()
    print(args)

    # keval 집계
    keval_scores = []
    for file in Path(args.keval).glob('*__keval.jsonl'):
        model, testset = file.stem.split('__')[:2]
        df = pd.read_json(file, orient='records', lines=True)
        df = pd.json_normalize(df['pairs'].explode())

        keval = df['keval'].str.extract(r"\[\[(\d+\.?\d*)\]\]")[0]
        # backup
        keval[keval.isna()] = df.loc[keval.isna(), 'keval'].str.extract(r"\[(\d+\.?\d*)\]")[0]
    
        keval = keval.dropna().astype(float)
        score = keval.mean() / 10
        count = keval.count()

        keval_scores.append({
            'testset': testset,
            'model': model,
            'keval_score': score,
            'keval': f"{score:.2f} ({count})"
        })
    keval_score_df = pd.DataFrame(keval_scores)

    # kgrammar 집계
    kgrammar_scores = []
    for file in Path(args.kgrammar).glob('*__kgrammar.jsonl'):
        model, testset = file.stem.split('__')[:2]
        df = pd.read_json(file, orient='records', lines=True)
        df = pd.json_normalize(df['pairs'].explode())
        
        kgrammar = df['kgrammar'].str.extract(r"<wrong count>\s*(\d+)\s*</wrong count>")[0]
        kgrammar = kgrammar.dropna().astype(int)
        
        score = (kgrammar == 0).mean()
        count = kgrammar.count()

        kgrammar_scores.append({
            'testset': testset,
            'model': model,
            'kgrammar_score': score,
            'kgrammar': f"{score:.2f} ({count})"
        })
    kgrammar_score_df = pd.DataFrame(kgrammar_scores)

    # 통합
    score_df = pd.merge(keval_score_df, kgrammar_score_df, on=['testset', 'model'])
    score_df['average'] = score_df[['keval_score', 'kgrammar_score']].mean(axis=1)
    score_df = score_df[['testset', 'model', 'average', 'keval', 'kgrammar']]
    
    # 출력
    for testset, df in score_df.groupby(by='testset', sort=False):
        print("\n\n# Testset:", testset, '\n')
        print(df.drop(columns='testset').sort_values(by='average', ascending=False).to_markdown(floatfmt='.2f'))


if __name__ == "__main__":
    main()