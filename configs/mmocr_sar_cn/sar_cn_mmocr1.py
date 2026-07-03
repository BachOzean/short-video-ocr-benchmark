from pathlib import Path

import mmocr

MMOCR_ROOT = Path(mmocr.__file__).resolve().parent
CONFIG_DIR = Path(__file__).resolve().parent

_base_ = [
    str(MMOCR_ROOT / '.mim/configs/textrecog/_base_/default_runtime.py'),
    str(MMOCR_ROOT / '.mim/configs/textrecog/sar/_base_sar_resnet31_parallel-decoder.py'),
]

dictionary = dict(
    type='Dictionary',
    dict_file=str(CONFIG_DIR / 'dict_printed_chinese_english_digits.txt'),
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=True,
    with_unknown=True)

model = dict(
    decoder=dict(
        dictionary=dictionary,
        max_seq_len=30))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RescaleToHeight', height=48, min_width=48, max_width=256, width_divisor=4),
    dict(type='PadToWidth', width=256),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='PackTextRecogInputs', meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
