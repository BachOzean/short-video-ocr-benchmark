# OCR算法评测结果分析

## 1. 评测口径

本次评测目标是比较多种 OCR 算法在短剧视频全画面识别任务上的效果。评测输入为按 0.5s/frame 抽取的视频帧，所有算法 raw output 均先经过同一套 subtitle-level merge 后处理，再与人工标注字幕真值计算字符级指标。

正式测试集：25 条短剧视频，均存在非空 `ocr_gt`，真值总字符数为 5,407。

统一后处理配置：

| 后处理项 | 配置 |
|---|---|
| merge 类型 | adjacent conservative fuzzy merge |
| 帧间隔 | 0.5s/frame |
| 相似度阈值 | 0.88 |
| 短文本相似度阈值 | 0.94 |
| 最大空白间隔 | 1.0s |

评价指标：

| 指标 | 含义 |
|---|---|
| CER | 字符错误率，越低越好 |
| Precision | 预测字符中正确字符占比，反映误识别和多识别问题 |
| Recall | 真值字符中被正确识别的占比，反映漏识别问题 |
| F1 | Precision 和 Recall 的调和平均 |

## 2. 总体结果

以下为 micro-average 结果，即在全测试集字符级汇总后计算。

| 排名（CER） | 方法 | CER ↓ | Precision ↑ | Recall ↑ | F1 ↑ |
|---:|---|---:|---:|---:|---:|
| 1 | GoMatching++ | 0.5626 | 0.6409 | 0.7202 | 0.6782 |
| 2 | RapidOCR | 0.7862 | 0.5520 | 0.8839 | 0.6796 |
| 3 | PPOCR v4 | 0.9758 | 0.4986 | 0.9014 | 0.6420 |
| 4 | PPOCR v5 | 1.0832 | 0.4686 | 0.8944 | 0.6150 |
| 5 | EasyOCR | 1.0891 | 0.4418 | 0.8051 | 0.5705 |
| 6 | Paddle-VL1.5 / PPOCR旧结果 | 1.1087 | 0.4624 | 0.8787 | 0.6060 |
| 7 | PPOCR v6 | 1.1087 | 0.4624 | 0.8787 | 0.6060 |
| 8 | GLM-OCR | 1.4825 | 0.3555 | 0.7363 | 0.4795 |
| 9 | MMOCR 工程配置：PANet_IC15 + SAR_CN | 19.8879 | 0.0319 | 0.6560 | 0.0609 |
| 10 | MMOCR 质量配置：DBNet++ / DBPP_r50 + SAR_CN | 25.3159 | 0.0231 | 0.5977 | 0.0444 |

VimTS 原配置本次未纳入有效排名，原因是本地暂未获得可运行的 VimTS 输出文件。

## 3. 新增 PPOCR v4/v5/v6 结论

1. PPOCR v4 是新增三版里最优。

PPOCR v4 的 CER 为 0.9758，优于 EasyOCR、PPOCR v5、PPOCR v6、GLM-OCR 和 MMOCR。它的 Recall 达到 0.9014，是当前所有有效算法中最高的，说明字幕召回能力很强。

2. PPOCR v4 的主要问题仍然是 Precision 不够高。

PPOCR v4 Precision 为 0.4986，低于 GoMatching++ 和 RapidOCR。这说明 v4 虽然能识别出更多真值字幕，但仍会输出较多额外字符，插入错误偏高。

3. PPOCR v5 相比 v4 退化。

PPOCR v5 的 CER 为 1.0832，比 v4 更差；Recall 仍然较高，为 0.8944，但 Precision 下降到 0.4686，说明额外字符更多。

4. PPOCR v6 与旧 Paddle-VL1.5 / PPOCR 结果完全一致。

PPOCR v6 的 micro 指标与旧 `paddle_vl15` 结果一致：CER 1.1087，Precision 0.4624，Recall 0.8787，F1 0.6060。需要确认这两份结果是否来自同一批 raw output 或同一模型配置。

## 4. 关键结论

1. GoMatching++ 仍是当前综合最稳方案。

GoMatching++ 的 CER 最低，为 0.5626，Precision 最高，为 0.6409。它的输出文本最干净，误识别和多识别最少，适合作为当前主推荐方案。

2. RapidOCR 的综合表现仍然很强。

RapidOCR 的 CER 排第二，为 0.7862；F1 为 0.6796，略高于 GoMatching++。它的 Recall 为 0.8839，说明召回能力强，但 Precision 低于 GoMatching++。

3. PPOCR v4 是新增结果中最值得保留的版本。

PPOCR v4 的 Recall 最高，CER 排第三。如果后续能通过字幕区域过滤、重复过滤或置信度过滤提升 Precision，它有机会接近 RapidOCR。

4. MMOCR 两套配置虽然已经成功出分，但不适合当前 full-frame 评测口径。

MMOCR 的两个配置分数极差，核心原因不是没有跑通，而是 full-frame 输入下检测出了大量非目标文字框或无效文字框，SAR_CN 又产生大量重复乱码，导致插入字符数爆炸。

| MMOCR配置 | 预测字符数 | GT字符数 | 主要问题 |
|---|---:|---:|---|
| DBNet++ / DBPP_r50 + SAR_CN | 140,115 | 5,407 | 过检测、过识别、插入极多 |
| PANet_IC15 + SAR_CN | 111,081 | 5,407 | 过检测、空 crop、插入极多 |

## 5. 指标排序

### CER 排序

1. GoMatching++：0.5626
2. RapidOCR：0.7862
3. PPOCR v4：0.9758
4. PPOCR v5：1.0832
5. EasyOCR：1.0891
6. Paddle-VL1.5 / PPOCR旧结果：1.1087
7. PPOCR v6：1.1087
8. GLM-OCR：1.4825
9. MMOCR PANet + SAR_CN：19.8879
10. MMOCR DBNet++ + SAR_CN：25.3159

### Precision 排序

1. GoMatching++：0.6409
2. RapidOCR：0.5520
3. PPOCR v4：0.4986
4. PPOCR v5：0.4686
5. Paddle-VL1.5 / PPOCR旧结果：0.4624
6. PPOCR v6：0.4624
7. EasyOCR：0.4418
8. GLM-OCR：0.3555
9. MMOCR PANet + SAR_CN：0.0319
10. MMOCR DBNet++ + SAR_CN：0.0231

### Recall 排序

1. PPOCR v4：0.9014
2. PPOCR v5：0.8944
3. RapidOCR：0.8839
4. Paddle-VL1.5 / PPOCR旧结果：0.8787
5. PPOCR v6：0.8787
6. EasyOCR：0.8051
7. GLM-OCR：0.7363
8. GoMatching++：0.7202
9. MMOCR PANet + SAR_CN：0.6560
10. MMOCR DBNet++ + SAR_CN：0.5977

### F1 排序

1. RapidOCR：0.6796
2. GoMatching++：0.6782
3. PPOCR v4：0.6420
4. PPOCR v5：0.6150
5. Paddle-VL1.5 / PPOCR旧结果：0.6060
6. PPOCR v6：0.6060
7. EasyOCR：0.5705
8. GLM-OCR：0.4795
9. MMOCR PANet + SAR_CN：0.0609
10. MMOCR DBNet++ + SAR_CN：0.0444

## 6. 推荐方案

### 当前主推荐

GoMatching++ 作为当前主方案。

理由：CER 最低、Precision 最高、输出文本最干净，后处理压力最小。

### 高召回备选

RapidOCR 和 PPOCR v4。

RapidOCR 的 F1 最高且 CER 排第二；PPOCR v4 的 Recall 最高，适合需要尽量减少漏识别的场景。

### 后续优化方向

1. 对 PPOCR v4 做额外过滤，重点提升 Precision。
2. 对 RapidOCR 做更严格去重和插入过滤，进一步降低 CER。
3. 对所有方法尝试字幕区域过滤，减少背景文字进入 merge。
4. MMOCR 若继续实验，必须先加入检测框过滤或字幕区域约束，否则 full-frame 输出不可用。
