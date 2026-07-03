#!/usr/bin/env python3
import csv
import json
from pathlib import Path

PRED = Path('/home/derbach/code/ocr_eval/results/easyocr_predictions.jsonl')
OUT_DIR = Path('/home/derbach/code/ocr_eval/results')

MODES = {
    'adjacent_exact': {'mode': 'exact'},
    'adjacent_sim_090': {'mode': 'similar', 'line_sim': 0.90, 'coverage': 0.90},
    'adjacent_sim_085': {'mode': 'similar', 'line_sim': 0.85, 'coverage': 0.85},
}


def metric_norm(s: str) -> str:
    out = []
    for ch in str(s):
        if '\u4e00' <= ch <= '\u9fff' or (ch.isascii() and ch.isalnum()):
            out.append(ch.lower())
    return ''.join(out)


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
        'cer': (s + d + ins) / n if n else 0.0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def unique_keep_order(raw_lines):
    out = []
    seen = set()
    for raw in raw_lines:
        n = metric_norm(raw)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(raw)
    return out


def frame_lines(frame):
    raw = [str(x).strip() for x in frame.get('texts', []) if str(x).strip()]
    normed = [metric_norm(x) for x in raw]
    pairs = [(r, n) for r, n in zip(raw, normed) if n]
    return pairs


def exact_duplicate(prev_pairs, cur_pairs):
    return '|'.join(n for _, n in prev_pairs) == '|'.join(n for _, n in cur_pairs)


def coverage_score(src, dst, line_sim):
    total = sum(len(n) for _, n in src)
    if total == 0:
        return 1.0
    covered = 0
    dst_norms = [n for _, n in dst]
    for _, n in src:
        if dst_norms and max(sim(n, d) for d in dst_norms) >= line_sim:
            covered += len(n)
    return covered / total


def similar_duplicate(prev_pairs, cur_pairs, cfg):
    if not prev_pairs or not cur_pairs:
        return exact_duplicate(prev_pairs, cur_pairs)
    a = coverage_score(cur_pairs, prev_pairs, cfg['line_sim'])
    b = coverage_score(prev_pairs, cur_pairs, cfg['line_sim'])
    return min(a, b) >= cfg['coverage']


def dedup_record(record, cfg):
    kept_frames = []
    skipped_frames = 0
    prev_kept = None
    for frame in record.get('frame_texts', []):
        pairs = frame_lines(frame)
        if not pairs:
            continue
        dup = False
        if prev_kept is not None:
            if cfg['mode'] == 'exact':
                dup = exact_duplicate(prev_kept, pairs)
            else:
                dup = similar_duplicate(prev_kept, pairs, cfg)
        if dup:
            skipped_frames += 1
            continue
        kept_frames.append({'t': frame.get('t', 0.0), 'texts': [r for r, _ in pairs], 'norms': [n for _, n in pairs]})
        prev_kept = pairs

    raw_lines = []
    for frame in kept_frames:
        raw_lines.extend(frame['texts'])
    pred = unique_keep_order(raw_lines)
    return pred, kept_frames, skipped_frames


def summarize(per_video):
    totals = {'gt_chars': 0, 'pred_chars': 0, 'correct': 0, 'substitutions': 0, 'deletions': 0, 'insertions': 0, 'frames': 0, 'kept_frames': 0, 'skipped_frames': 0}
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


def run_mode(name, cfg, records):
    pred_path = OUT_DIR / f'easyocr_{name}_predictions.jsonl'
    metrics_path = OUT_DIR / f'easyocr_{name}_metrics.json'
    csv_path = OUT_DIR / f'easyocr_{name}_per_video.csv'
    per_video = []
    with pred_path.open('w', encoding='utf-8') as f:
        for r in records:
            pred, kept_frames, skipped_frames = dedup_record(r, cfg)
            m = doc_metrics(r.get('gt_lines', []), pred)
            item = {
                'engine': 'easyocr',
                'dedup_mode': name,
                'video_id': r['video_id'],
                'gt_lines': r.get('gt_lines', []),
                'pred_lines': pred,
                'kept_frames_detail': kept_frames,
                'frames': r.get('frames', 0),
                'kept_frames': len(kept_frames),
                'skipped_frames': skipped_frames,
                **m,
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            per_video.append({k: item[k] for k in ['video_id', 'frames', 'kept_frames', 'skipped_frames', 'gt_chars', 'pred_chars', 'correct', 'substitutions', 'deletions', 'insertions', 'cer', 'precision', 'recall', 'f1']})
    micro, macro = summarize(per_video)
    result = {
        'mode': name,
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
        fields = ['video_id', 'frames', 'kept_frames', 'skipped_frames', 'gt_chars', 'pred_chars', 'correct', 'substitutions', 'deletions', 'insertions', 'cer', 'precision', 'recall', 'f1']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_video)
    return result


def main():
    records = [json.loads(line) for line in PRED.read_text(encoding='utf-8').splitlines() if line.strip()]
    results = [run_mode(name, cfg, records) for name, cfg in MODES.items()]
    comparison = []
    for r in results:
        comparison.append({
            'mode': r['mode'],
            'cer': r['micro']['cer'],
            'precision': r['micro']['precision'],
            'recall': r['micro']['recall'],
            'f1': r['micro']['f1'],
            'frames': r['micro']['frames'],
            'kept_frames': r['micro']['kept_frames'],
            'skipped_frames': r['micro']['skipped_frames'],
            'pred_chars': r['micro']['pred_chars'],
            'insertions': r['micro']['insertions'],
            'deletions': r['micro']['deletions'],
            'substitutions': r['micro']['substitutions'],
        })
    path = OUT_DIR / 'easyocr_adjacent_frame_dedup_comparison.json'
    path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'comparison_file': str(path), 'comparison': comparison}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
