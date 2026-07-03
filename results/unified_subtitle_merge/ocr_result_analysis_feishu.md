# OCR算法评测结果分析

## 1. 评测背景

本次评测目标是比较多种 OCR 算法在短剧视频全画面识别任务上的效果。评测输入为按 0.5s/frame 抽取的视频帧，算法输出经过统一的字幕级后处理，再与人工标注字幕真值进行字符级对齐计算。

正式测试集仅包含存在非空 `ocr_gt` 的样本，共 25 条短剧视频，真值总字符数为 5,407。

## 2. 评测口径

所有算法统一使用同一套 subtitle-level merge 后处理：

| 后处理项 | 配置 |
|---|---|
| merge 类型 | adjacent conservative fuzzy merge |
| 帧间隔 | 0.5s/frame |
| 相似度阈值 | 0.88 |
| 短文本相似度阈值 | 0.94 |
| 最大空白间隔 | 1.0s |

评价指标统一按字符统计，单个汉字、英文字母、数字均计为 1 个字符。

| 指标 | 含义 |
|---|---|
| CER | 字符错误率，越低越好 |
| Precision | 预测字符中正确字符占比，反映误识别和多识别问题 |
| Recall | 真值字符中被正确识别的占比，反映漏识别问题 |
| F1 | Precision 和 Recall 的调和平均 |

## 3. 总体结果

以下为 micro-average 结果，即在全测试集字符级汇总后计算。

| 排名（CER） | 方法 | CER ↓ | Precision ↑ | Recall ↑ | F1 ↑ |
|---:|---|---:|---:|---:|---:|
| 1 | GoMatching++ | 0.5626 | 0.6409 | 0.7202 | 0.6782 |
| 2 | RapidOCR | 0.7862 | 0.5520 | 0.8839 | 0.6796 |
| 3 | EasyOCR | 1.0891 | 0.4418 | 0.8051 | 0.5705 |
| 4 | Paddle-VL1.5 / PPOCR结果 | 1.1087 | 0.4624 | 0.8787 | 0.6060 |
| 5 | GLM-OCR | 1.4825 | 0.3555 | 0.7363 | 0.4795 |
| 6 | MMOCR 工程配置：PANet_IC15 + SAR_CN | 19.8879 | 0.0319 | 0.6560 | 0.0609 |
| 7 | MMOCR 质量配置：DBNet++ / DBPP_r50 + SAR_CN | 25.3159 | 0.0231 | 0.5977 | 0.0444 |

VimTS 原配置本次未纳入有效排名，原因是本地暂未获得可运行的 VimTS 输出文件。

## 4. 关键结论

1. GoMatching++ 是本轮最稳的方案。

GoMatching++ 的 CER 最低，为 0.5626，Precision 最高，为 0.6409。它的主要优势是误识别和多识别更少，整体错误量最可控。虽然 Recall 不是最高，但综合 CER 表现最好，适合作为当前主推荐方案。

2. RapidOCR 的召回最好，F1 略高于 GoMatching++。

RapidOCR 的 Recall 达到 0.8839，是所有有效算法中最高的，说明它更容易把字幕内容识别出来。但 Precision 只有 0.5520，插入字符较多，因此 CER 高于 GoMatching++。如果业务更关注“尽量不要漏字幕”，RapidOCR 有优势；如果更关注最终文本干净程度，GoMatching++ 更合适。

3. EasyOCR 和 Paddle-VL1.5 / PPOCR 结果都存在明显多识别问题。

EasyOCR 的 Recall 为 0.8051，说明有一定识别能力，但 Precision 只有 0.4418，CER 超过 1。Paddle-VL1.5 / PPOCR 的 Recall 更高，为 0.8787，但 Precision 也偏低，为 0.4624。两者的问题都不是完全识别不到，而是识别出了大量额外字符，导致插入错误偏高。

4. GLM-OCR 在当前设置下整体效果偏弱。

GLM-OCR 的 CER 为 1.4825，Precision 只有 0.3555，F1 为 0.4795。它相比 EasyOCR / Paddle-VL1.5 没有体现出明显优势，主要问题同样是额外文本较多，同时有效字幕召回也不占优。

5. MMOCR 两套配置虽然已经成功出分，但不适合当前 full-frame 评测口径。

MMOCR 的两个配置分数极差，核心原因不是没有跑通，而是 full-frame 输入下检测出了大量非目标文字框或无效文字框，SAR_CN 又产生大量重复乱码，导致插入字符数爆炸。

| MMOCR配置 | 预测字符数 | GT字符数 | 主要问题 |
|---|---:|---:|---|
| DBNet++ / DBPP_r50 + SAR_CN | 140,115 | 5,407 | 过检测、过识别、插入极多 |
| PANet_IC15 + SAR_CN | 111,081 | 5,407 | 过检测、空 crop、插入极多 |

因此，MMOCR 当前不能直接作为全画面字幕 OCR 方案使用。如果后续要继续尝试，需要先做检测框过滤或字幕区域约束，否则模型输出会被大量背景文本和乱码污染。

## 5. 按指标观察

### 5.1 CER 维度

CER 最低的是 GoMatching++，说明其总体字符错误最少。RapidOCR 排第二，但与 GoMatching++ 有明显差距。MMOCR 两个配置的 CER 极高，主要由插入字符导致，不具备可比性。

CER 排序：

1. GoMatching++：0.5626
2. RapidOCR：0.7862
3. EasyOCR：1.0891
4. Paddle-VL1.5 / PPOCR结果：1.1087
5. GLM-OCR：1.4825
6. MMOCR PANet + SAR_CN：19.8879
7. MMOCR DBNet++ + SAR_CN：25.3159

### 5.2 Precision 维度

Precision 最高的是 GoMatching++，说明它输出的字符更干净，误识别和多识别更少。RapidOCR、EasyOCR、Paddle-VL1.5 虽然能识别出不少字幕，但都存在较明显的额外字符问题。

Precision 排序：

1. GoMatching++：0.6409
2. RapidOCR：0.5520
3. Paddle-VL1.5 / PPOCR结果：0.4624
4. EasyOCR：0.4418
5. GLM-OCR：0.3555
6. MMOCR PANet + SAR_CN：0.0319
7. MMOCR DBNet++ + SAR_CN：0.0231

### 5.3 Recall 维度

RapidOCR 召回最高，Paddle-VL1.5 / PPOCR 也接近 RapidOCR。说明这类 OCR 更容易把字幕字符覆盖到，但代价是输出更脏。GoMatching++ 的 Recall 不是最高，但 Precision 更好，因此 CER 更优。

Recall 排序：

1. RapidOCR：0.8839
2. Paddle-VL1.5 / PPOCR结果：0.8787
3. EasyOCR：0.8051
4. GLM-OCR：0.7363
5. GoMatching++：0.7202
6. MMOCR PANet + SAR_CN：0.6560
7. MMOCR DBNet++ + SAR_CN：0.5977

### 5.4 F1 维度

RapidOCR 的 F1 为 0.6796，略高于 GoMatching++ 的 0.6782。但这个差异非常小，而且 RapidOCR 的 CER 更高、Precision 更低。因此综合选择时不建议只看 F1，需要结合 CER 和 Precision。

F1 排序：

1. RapidOCR：0.6796
2. GoMatching++：0.6782
3. Paddle-VL1.5 / PPOCR结果：0.6060
4. EasyOCR：0.5705
5. GLM-OCR：0.4795
6. MMOCR PANet + SAR_CN：0.0609
7. MMOCR DBNet++ + SAR_CN：0.0444

## 6. 方法级分析

### GoMatching++

优势：

- CER 最低，整体错误最少。
- Precision 最高，输出文本相对干净。
- 更适合作为当前阶段的主方案或 baseline。

不足：

- Recall 低于 RapidOCR / Paddle-VL1.5 / EasyOCR，存在一定漏识别。
- 如果业务更看重“召回所有可能字幕”，需要补召回策略。

### RapidOCR

优势：

- Recall 最高，字幕覆盖能力强。
- F1 略高于 GoMatching++。

不足：

- Precision 明显低于 GoMatching++，存在较多额外字符。
- CER 高于 GoMatching++，说明插入错误影响较大。

适用判断：

- 如果目标是高召回，可以考虑 RapidOCR。
- 如果目标是最终字幕文本质量，RapidOCR 需要更强的去噪和 merge 过滤。

### EasyOCR

优势：

- Recall 仍有 0.8051，说明不是完全失效。

不足：

- Precision 偏低，CER 超过 1。
- 错误多来自误识别和插入字符。

适用判断：

- 当前不建议作为主方案。
- 若继续使用，需要针对字幕区域、文本置信度和重复文本做更强过滤。

### Paddle-VL1.5 / PPOCR结果

优势：

- Recall 高，为 0.8787，接近 RapidOCR。
- F1 高于 EasyOCR。

不足：

- Precision 偏低，CER 仍超过 1。
- 输出额外字符较多，最终文本不够干净。

适用判断：

- 可作为高召回备选。
- 需要后处理控制插入错误，否则最终 CER 不理想。

### GLM-OCR

优势：

- 有一定字幕识别能力。

不足：

- CER、Precision、F1 都明显弱于 GoMatching++ / RapidOCR。
- 当前 full-frame 设置下没有体现明显优势。

适用判断：

- 当前不建议优先投入。

### MMOCR：DBNet++ / PANet + SAR_CN

优势：

- 已经解决环境和兼容性问题，能够跑通并报分。

不足：

- full-frame 输入下严重过检测和过识别。
- 插入字符数量远超真值字符数量。
- PANet 还存在部分 invalid crop，需要 fallback。

适用判断：

- 不建议直接用于当前 full-frame OCR 评测。
- 若继续探索，必须先加入字幕区域定位、检测框尺寸过滤、置信度过滤、重复框过滤等策略。

## 7. 推荐方案

### 当前主推荐

GoMatching++ 作为当前主方案。

理由：

- CER 最低。
- Precision 最高。
- 输出文本最干净，后处理压力相对最小。

### 高召回备选

RapidOCR 作为高召回备选方案。

理由：

- Recall 最高。
- F1 与 GoMatching++ 基本持平。
- 适合需要尽量捕捉所有字幕的场景。

### 不建议继续优先投入

- MMOCR 当前 full-frame 方案不建议继续作为主线，除非先做字幕区域约束。
- GLM-OCR 当前效果不优先。
- EasyOCR / Paddle-VL1.5 可以保留作参考，但需要强后处理才能接近可用。

## 8. 下一步建议

1. 以 GoMatching++ 为主 baseline，抽查高 CER 视频，分析漏识别来源。

2. 对 RapidOCR 做更严格的去重和插入过滤，尝试提升 Precision，同时保持高 Recall。

3. 对所有方法增加字幕区域过滤策略，例如只保留画面下方或高置信字幕区域，减少背景文字进入 merge。

4. 对 MMOCR 若继续实验，应先做检测框过滤，而不是直接 full-frame 检测后识别。

5. 补齐 VimTS 可运行输出后，再纳入同一套统一 merge/eval 表格比较。
