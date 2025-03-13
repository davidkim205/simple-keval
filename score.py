import pandas as pd
import argparse

from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keval', type=str, default='results_keval/')
    parser.add_argument('--kgrammar', type=str, default='results_kgrammar/')
    args = parser.parse_args()
    print(args)

    # keval 집계
    keval_scores = []
    for file in Path(args.keval).glob('*__keval.jsonl'):
        model = '__'.join(file.stem.split('__')[:2])
        df = pd.read_json(file, orient='records', lines=True)
        df = pd.json_normalize(df['pairs'].explode())

        keval = df['keval'].str.extract(r"\[\[(\d+\.?\d*)\]\]")[0]
        # backup
        keval[keval.isna()] = df.loc[keval.isna(), 'keval'].str.extract(r"\[(\d+\.?\d*)\]")[0]
    
        keval = keval.dropna().astype(float)
        score = keval.mean() / 10
        count = keval.count()

        keval_scores.append({
            'model': model,
            'keval_score': score,
            'keval': f"{score:.2f} ({count})"
        })
    keval_score_df = pd.DataFrame(keval_scores)

    # kgrammar 집계
    kgrammar_scores = []
    for file in Path(args.kgrammar).glob('*__kgrammar.jsonl'):
        model = '__'.join(file.stem.split('__')[:2])
        df = pd.read_json(file, orient='records', lines=True)
        df = pd.json_normalize(df['pairs'].explode())
        
        kgrammar = df['kgrammar'].str.extract(r"<wrong count>\s*(\d+)\s*</wrong count>")[0]
        kgrammar = kgrammar.dropna().astype(int)
        
        score = (kgrammar == 0).mean()
        count = kgrammar.count()

        kgrammar_scores.append({
            'model': model,
            'kgrammar_score': score,
            'kgrammar': f"{score:.2f} ({count})"
        })
    kgrammar_score_df = pd.DataFrame(kgrammar_scores)

    # 통합
    score_df = pd.merge(keval_score_df, kgrammar_score_df, on='model')
    score_df['average'] = score_df[['keval_score', 'kgrammar_score']].mean(axis=1)
    score_df = score_df[['model', 'average', 'keval', 'kgrammar']]

    print(score_df.sort_values(by='average', ascending=False).to_markdown(index=False, floatfmt='.2f'))


if __name__ == "__main__":
    main()