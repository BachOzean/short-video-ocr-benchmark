#!/usr/bin/env python3
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PRED = Path('/home/derbach/code/ocr_eval/results/easyocr_predictions.jsonl')
OUT_DIR = Path('/home/derbach/code/ocr_eval/results')

STRATEGIES = {
    'conservative': {
        'long': 0.86,
        'mid': 0.90,
        'short': 0.95,
        'drop_len1_count_lt': 2,
        'drop_ascii_len1_count_lt': 3,
    },
    'balanced': {
        'long': 0.78,
        'mid': 0.84,
        'short': 0.92,
        'drop_len1_count_lt': 2,
        'drop_ascii_len1_count_lt': 2,
    },
    'aggressive': {
        'long': 0.70,
        'mid': 0.78,
        'short': 0.88,
        'drop_len1_count_lt': 2,
        'drop_ascii_len1_count_lt': 2,
    },
}


def metric_norm(s: str) -> str:
    out = []
    for ch in str(s):
        if '\u4e00' <= ch <= '\u9fff' or (ch.isascii() and ch.isalnum()):
            out.append(ch.lower())
    return ''.join(out)


def loose_norm(s: str) -> str:
    return metric_norm(s)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - edit_distance(a, b) / max(len(a), len(b), 1)


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
            best = min(best, (dp[i - 1][j] + 1, 'D'), (dp[i][j - 1] + 1, 'I'), key=lambda x: x[0])
            dp[i][j], bt[i][j] = best
    i, j = n, m
    c = s = d = ins = 0
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


def doc_metrics(gt_lines, pred_lines):
    gt = metric_norm(''.join(gt_lines))
    pred = metric_norm(''.join(pred_lines))
    c, s, d, ins = edit_ops(gt, pred)
    n = len(gt)
    cer = (s + d + ins) / n if n else 0.0
    precision = c / (c + s + ins) if (c + s + ins) else 0.0
    recall = c / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'gt_chars': n,
        'pred_chars': len(pred),
        'correct': c,
        'substitutions': s,
        'deletions': d,
        'insertions': ins,
        'cer': cer,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def threshold_for(a: str, b: str, cfg: dict) -> float:
    mn = min(len(a), len(b))
    mx = max(len(a), len(b))
    if mn <= 2:
        return cfg['short']
    if mx <= 8:
        return cfg['mid']
    return cfg['long']


def cluster_same(a: str, b: str, cfg: dict) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) <= 1:
        return False
    return sim(a, b) >= threshold_for(a, b, cfg)


def pick_raw(raw_counter: Counter) -> str:
    # Prefer the most frequent raw. On ties, prefer the one with more evaluable chars.
    return sorted(raw_counter.items(), key=lambda kv: (-kv[1], -len(metric_norm(kv[0])), kv[0]))[0][0]


def dedup_record(record: dict, cfg: dict):
    occurrences = []
    first_seen = {}
    raw_by_norm = defaultdict(Counter)
    count_by_norm = Counter()

    seq = 0
    for frame in record.get('frame_texts', []):
        t = frame.get('t', 0.0)
        for raw in frame.get('texts', []):
            n = loose_norm(raw)
            if not n:
                continue
            occurrences.append((seq, t, raw, n))
            count_by_norm[n] += 1
            raw_by_norm[n][raw] += 1
            first_seen.setdefault(n, (seq, t))
            seq += 1

    entries = []
    for n, cnt in count_by_norm.items():
        if len(n) == 1 and cnt < cfg['drop_len1_count_lt']:
            continue
        if len(n) == 1 and n.isascii() and cnt < cfg['drop_ascii_len1_count_lt']:
            continue
        entries.append({
            'norm': n,
            'raw': pick_raw(raw_by_norm[n]),
            'count': cnt,
            'first_seq': first_seen[n][0],
            'first_t': first_seen[n][1],
        })

    # Keep temporal order as the primary structure, but let stronger repeated variants become cluster representatives.
    entries.sort(key=lambda e: (e['first_seq'], -e['count'], -len(e['norm'])))
    clusters = []
    for e in entries:
        best_i = None
        best_s = -1.0
        for i, cl in enumerate(clusters):
            # Compare to representative and a few strong members to avoid bad early representative lock-in.
            candidates = [cl['rep_norm']] + [m['norm'] for m in cl['members'][:4]]
            local_best = max(sim(e['norm'], c) for c in candidates)
            if local_best > best_s and any(cluster_same(e['norm'], c, cfg) for c in candidates):
                best_s = local_best
                best_i = i
        if best_i is None:
            clusters.append({
                'rep_norm': e['norm'],
                'rep_raw': e['raw'],
                'count': e['count'],
                'first_seq': e['first_seq'],
                'first_t': e['first_t'],
                'members': [e],
            })
        else:
            cl = clusters[best_i]
            cl['members'].append(e)
            cl['count'] += e['count']
            cl['members'].sort(key=lambda m: (-m['count'], -len(m['norm']), m['first_seq']))
            rep = cl['members'][0]
            cl['rep_norm'] = rep['norm']
            cl['rep_raw'] = rep['raw']
            cl['first_seq'] = min(cl['first_seq'], e['first_seq'])
            cl['first_t'] = min(cl['first_t'], e['first_t'])

    clusters.sort(key=lambda cl: cl['first_seq'])
    pred = [cl['rep_raw'] for cl in clusters]
    debug = [{
        'text': cl['rep_raw'],
        'norm': cl['rep_norm'],
        'count': cl['count'],
        'first_t': round(cl['first_t'], 3),
        'variants': len(cl['members']),
        'variant_texts': [m['raw'] for m in cl['members'][:8]],
    } for cl in clusters]
    return pred, debug


def summarize(per_video):
    totals = {'gt_chars': 0, 'pred_chars': 0, 'correct': 0, 'substitutions': 0, 'deletions': 0, 'insertions': 0}
    for x in per_video:
        for k in totals:
            totals[k] += x[k]
    c = totals['correct']
    s = totals['substitutions']
    d = totals['deletions']
    ins = totals['insertions']
    n = totals['gt_chars']
    micro = {
        **totals,
        'videos': len(per_video),
        'cer': (s + d + ins) / n if n else 0.0,
        'precision': c / (c + s + ins) if (c + s + ins) else 0.0,
        'recall': c / n if n else 0.0,
    }
    micro['f1'] = 2 * micro['precision'] * micro['recall'] / (micro['precision'] + micro['recall']) if (micro['precision'] + micro['recall']) else 0.0
    macro = {f'macro_{k}': sum(x[k] for x in per_video) / len(per_video) for k in ['cer', 'precision', 'recall', 'f1']}
    return micro, macro


def write_outputs(name, records, cfg):
    pred_path = OUT_DIR / f'easyocr_dedup_{name}_predictions.jsonl'
    metrics_path = OUT_DIR / f'easyocr_dedup_{name}_metrics.json'
    csv_path = OUT_DIR / f'easyocr_dedup_{name}_per_video.csv'

    per_video = []
    with pred_path.open('w', encoding='utf-8') as f:
        for r in records:
            pred, debug = dedup_record(r, cfg)
            m = doc_metrics(r.get('gt_lines', []), pred)
            item = {
                'engine': 'easyocr',
                'dedup_strategy': name,
                'video_id': r['video_id'],
                'frames': r.get('frames', 0),
                'gt_lines': r.get('gt_lines', []),
                'pred_lines_dedup': pred,
                'clusters': debug,
                **m,
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            per_video.append({k: item[k] for k in ['video_id', 'frames', 'gt_chars', 'pred_chars', 'correct', 'substitutions', 'deletions', 'insertions', 'cer', 'precision', 'recall', 'f1']})

    micro, macro = summarize(per_video)
    result = {
        'strategy': name,
        'config': cfg,
        'normalization': 'keep CJK unified ideographs, ASCII letters, ASCII digits; lowercase ASCII; drop whitespace/punctuation/symbols',
        'micro': micro,
        'macro': macro,
        'prediction_file': str(pred_path),
        'per_video_csv': str(csv_path),
        'per_video': per_video,
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        fields = ['video_id', 'frames', 'gt_chars', 'pred_chars', 'correct', 'substitutions', 'deletions', 'insertions', 'cer', 'precision', 'recall', 'f1']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_video)
    return result


def main():
    records = [json.loads(line) for line in PRED.read_text(encoding='utf-8').splitlines() if line.strip()]
    results = [write_outputs(name, records, cfg) for name, cfg in STRATEGIES.items()]
    comparison = []
    for r in results:
        comparison.append({
            'strategy': r['strategy'],
            'cer': r['micro']['cer'],
            'precision': r['micro']['precision'],
            'recall': r['micro']['recall'],
            'f1': r['micro']['f1'],
            'pred_chars': r['micro']['pred_chars'],
            'insertions': r['micro']['insertions'],
            'deletions': r['micro']['deletions'],
            'substitutions': r['micro']['substitutions'],
        })
    comp_path = OUT_DIR / 'easyocr_dedup_comparison.json'
    comp_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'comparison_file': str(comp_path), 'comparison': comparison}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
