---
license: cc-by-nc-nd-4.0
---
---
{}
---
# Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges (Main EMNLP 2024)

## Introduction

This document presents the accompanying dataset for the paper titled **"Multi-Dialect Vietnamese: Task, Dataset, Baseline Models, and Challenges"**. The dataset, referred to as the **Vi**etnamese **M**ulti-**D**ialect (ViMD) dataset, is a comprehensive resource designed to capture the linguistic diversity represented by 63 provincial dialects spoken across Vietnam. The paper is available at https://aclanthology.org/2024.emnlp-main.426.

## Citation

If you use this paper and its dataset in your research, please cite it as follows:

```bibtex
@inproceedings{dinh-etal-2024-multi,
    title = "Multi-Dialect {V}ietnamese: Task, Dataset, Baseline Models and Challenges",
    author = "Dinh, Nguyen  and
      Dang, Thanh  and
      Thanh Nguyen, Luan  and
      Nguyen, Kiet",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.426",
    pages = "7476--7498",
    abstract = "Vietnamese, a low-resource language, is typically categorized into three primary dialect groups that belong to Northern, Central, and Southern Vietnam. However, each province within these regions exhibits its own distinct pronunciation variations. Despite the existence of various speech recognition datasets, none of them has provided a fine-grained classification of the 63 dialects specific to individual provinces of Vietnam. To address this gap, we introduce Vietnamese Multi-Dialect (ViMD) dataset, a novel comprehensive dataset capturing the rich diversity of 63 provincial dialects spoken across Vietnam. Our dataset comprises 102.56 hours of audio, consisting of approximately 19,000 utterances, and the associated transcripts contain over 1.2 million words. To provide benchmarks and simultaneously demonstrate the challenges of our dataset, we fine-tune state-of-the-art pre-trained models for two downstream tasks: (1) Dialect identification and (2) Speech recognition. The empirical results suggest two implications including the influence of geographical factors on dialects, and the constraints of current approaches in speech recognition tasks involving multi-dialect speech data. Our dataset is available for research purposes.",
}
```

## Overview of the Dataset

- **Source**: News programs from the broadcasting stations of the 63 provinces of Vietnam.
- **Overall Statistics**:

<table border="1" style="border-collapse: collapse; width: 100%; text-align: center;">
  <tr style="background-color: #f2f2f2;">
    <th rowspan="2" style="text-align: center;"></th>
    <th colspan="4" style="text-align: center;">Per Provincial Dialect</th>
    <th colspan="3" style="text-align: center;">Data Set</th>
    <th rowspan="2" style="text-align: center;">Total</th>
  </tr>
  <tr style="background-color: #f2f2f2;">
    <th style="text-align: center;">Min.</th>
    <th style="text-align: center;">Max.</th>
    <th style="text-align: center;">Mean</th>
    <th style="text-align: center;">Std.</th>
    <th style="text-align: center;">Train</th>
    <th style="text-align: center;">Valid.</th>
    <th style="text-align: center;">Test</th>
  </tr>
  <tr>
    <th style="text-align: center; background-color: #f2f2f2;">Duration</th>
    <td style="text-align: center;">89.11m</td>
    <td style="text-align: center;">117.98m</td>
    <td style="text-align: center;">97.68m</td>
    <td style="text-align: center;">4.18m</td>
    <td style="text-align: center;">81.43h</td>
    <td style="text-align: center;">10.26h</td>
    <td style="text-align: center;">10.87h</td>
    <td style="text-align: center;">102.56h</td>
  </tr>
  <tr>
    <th style="text-align: center; background-color: #f2f2f2;">#record</th>
    <td style="text-align: center;">263</td>
    <td style="text-align: center;">363</td>
    <td style="text-align: center;">301</td>
    <td style="text-align: center;">21</td>
    <td style="text-align: center;">15,023</td>
    <td style="text-align: center;">1,900</td>
    <td style="text-align: center;">2,026</td>
    <td style="text-align: center;">18,949</td>
  </tr>
  <tr>
    <th style="text-align: center; background-color: #f2f2f2;">#speaker</th>
    <td style="text-align: center;">88</td>
    <td style="text-align: center;">309</td>
    <td style="text-align: center;">206</td>
    <td style="text-align: center;">47</td>
    <td style="text-align: center;">10,291</td>
    <td style="text-align: center;">1,320</td>
    <td style="text-align: center;">1,344</td>
    <td style="text-align: center;">12,955</td>
  </tr>
  <tr>
    <th style="text-align: center; background-color: #f2f2f2;">#word</th>
    <td style="text-align: center;">17,038</td>
    <td style="text-align: center;">24,557</td>
    <td style="text-align: center;">19,669</td>
    <td style="text-align: center;">1,174</td>
    <td style="text-align: center;">981,391</td>
    <td style="text-align: center;">125,305</td>
    <td style="text-align: center;">132,471</td>
    <td style="text-align: center;">1,239,167</td>
  </tr>
  <tr>
    <th style="text-align: center; background-color: #f2f2f2;">#unique-word</th>
    <td style="text-align: center;">1,120</td>
    <td style="text-align: center;">1,639</td>
    <td style="text-align: center;">1,405</td>
    <td style="text-align: center;">103</td>
    <td style="text-align: center;">4,813</td>
    <td style="text-align: center;">2,660</td>
    <td style="text-align: center;">2,773</td>
    <td style="text-align: center;">5,155</td>
  </tr>
</table>


- **Attributes**:

<table border="1" style="border-collapse: collapse; width: 100%; text-align: left;">
  <tr>
    <th style="background-color: #f2f2f2; text-align: center;">Key</th>
    <th style="background-color: #f2f2f2; text-align: center;">Description</th>
  </tr>
  <tr>
    <td style="background-color: #f2f2f2; text-align: center;">set</td>
    <td>The set of audio: `{'train', 'valid', 'test'}`.</td>
  </tr>
  <tr>
    <td style="background-color: #f2f2f2; text-align: center;">filename</td>
    <td>The filename follows the syntax `{province code}_{Sequence Number of Audio}`.</td>
  </tr>
  <tr>
    <td style="background-color: #f2f2f2; text-align: center;">text</td>
    <td>Transcript of the audio.</td>
  </tr>
  <tr>
    <td style="background-color: #f2f2f2; text-align: center;">length</td>
    <td>Length of the audio in seconds.</td>
  </tr>
  <tr>
    <td style="background-color: #f2f2f2; text-align: center;">province</td>
    <td>The provincial dialect code.</td>
  </tr>
  <tr>
    <td style="background-color: #f2f2f2; text-align: center;">region</td>
    <td>The regional dialect: `{'North', 'Central', 'South'}`.</td>
  </tr>
  <tr>
    <td style="background-color: #f2f2f2; text-align: center;">speakerID</td>
    <td>The speaker identification code follows the syntax `spk_{province code}_{Sequence Number of Speaker}`.</td>
  </tr>
  <tr>
    <td style="background-color: #f2f2f2; text-align: center;">gender</td>
    <td>Gender of the speaker (0 represents female and 1 represents male).</td>
  </tr>
</table>


- For further statistics, please refer to the paper.