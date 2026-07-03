#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

PRED = Path('/home/derbach/code/ocr_eval/results/easyocr_predictions.jsonl')
OUT_JSON = Path('/home/derbach/code/ocr_eval/results/easyocr_doc_metrics.json')
OUT_CSV = Path('/home/derbach/code/ocr_eval/results/easyocr_doc_metrics_per_video.csv')

# 文档口径核心是按字符统计。这里保留汉字、英文字母、数字；去掉空白、标点、符号，避免标点格式差异主导结果。
# 如果你希望标点也计入字符，可以把 normalize_text 改成只去空白。
def normalize_text(s: str) -> str:
    s = str(s)
    chars = []
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff' or ch.isascii() and ch.isalnum():
            chars.append(ch.lower())
    return ''.join(chars)


def edit_ops(ref: str, hyp: str):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[''] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = 'D'
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = 'I'
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = (dp[i - 1][j - 1], 'C')
            else:
                best = (dp[i - 1][j - 1] + 1, 'S')
            cand_d = (dp[i - 1][j] + 1, 'D')
            cand_i = (dp[i][j - 1] + 1, 'I')
            best = min(best, cand_d, cand_i, key=lambda x: x[0])
            dp[i][j], bt[i][j] = best
    i, j = n, m
    s = d = ins = c = 0
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == 'C':
            c += 1
            i -= 1
            j -= 1
        elif op == 'S':
            s += 1
            i -= 1
            j -= 1
        elif op == 'D':
            d += 1
            i -= 1
        elif op == 'I':
            ins += 1
            j -= 1
        else:
            raise RuntimeError((i, j, op))
    return c, s, d, ins


def metrics_for(gt_lines, pred_lines):
    gt = normalize_text(''.join(gt_lines))
    pred = normalize_text(''.join(pred_lines))
    c, s, d, ins = edit_ops(gt, pred)
    n = len(gt)
    pred_n = len(pred)
    cer = (s + d + ins) / n if n else 0.0
    precision = c / (c + s + ins) if (c + s + ins) else 0.0
    recall = c / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'gt_chars': n,
        'pred_chars': pred_n,
        'correct': c,
        'substitutions': s,
        'deletions': d,
        'insertions': ins,
        'cer': cer,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Compute document-style OCR metrics.')
    parser.add_argument('--pred', type=Path, default=PRED)
    parser.add_argument('--out-json', type=Path, default=OUT_JSON)
    parser.add_argument('--out-csv', type=Path, default=OUT_CSV)
    return parser.parse_args()


def as_lines(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def main():
    args = parse_args()
    records = [json.loads(line) for line in args.pred.read_text(encoding='utf-8').splitlines() if line.strip()]
    per_video = []
    totals = {'gt_chars': 0, 'pred_chars': 0, 'correct': 0, 'substitutions': 0, 'deletions': 0, 'insertions': 0}
    for r in records:
        gt_lines = as_lines(r.get('gt_lines', r.get('ocr_gt', '')))
        pred_lines = as_lines(r.get('pred_lines_unique', r.get('pred_lines', [])))
        m = metrics_for(gt_lines, pred_lines)
        item = {'video_id': r['video_id'], 'frames': r.get('frames', 0), **m}
        per_video.append(item)
        for k in totals:
            totals[k] += item[k]

    S = totals['substitutions']
    D = totals['deletions']
    I = totals['insertions']
    C = totals['correct']
    N = totals['gt_chars']
    summary = {
        **totals,
        'videos': len(per_video),
        'cer': (S + D + I) / N if N else 0.0,
        'precision': C / (C + S + I) if (C + S + I) else 0.0,
        'recall': C / N if N else 0.0,
    }
    summary['f1'] = 2 * summary['precision'] * summary['recall'] / (summary['precision'] + summary['recall']) if (summary['precision'] + summary['recall']) else 0.0
    macro = {}
    for k in ['cer', 'precision', 'recall', 'f1']:
        macro[f'macro_{k}'] = sum(x[k] for x in per_video) / len(per_video) if per_video else 0.0
    result = {'normalization': 'keep CJK unified ideographs, ASCII letters, ASCII digits; lowercase ASCII; drop whitespace/punctuation/symbols', 'micro': summary, 'macro': macro, 'per_video': per_video}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with args.out_csv.open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['video_id', 'frames', 'gt_chars', 'pred_chars', 'correct', 'substitutions', 'deletions', 'insertions', 'cer', 'precision', 'recall', 'f1']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_video)
    print(json.dumps({'json': str(args.out_json), 'csv': str(args.out_csv), 'micro': summary, 'macro': macro}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
