
<h1 align="center"> <a href="https://arxiv.org/pdf/2510.19457">MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models</a></h1>
<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2510.19457-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2510.19457) [![Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Paper-MINED-blue)](https://huggingface.co/papers/2510.19457) [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MINED-blue)](https://huggingface.co/datasets/kailinjiang/MINED) [![code](https://img.shields.io/badge/Code-MINED-blue?logo=github)](https://github.com/MINED-LMM/MINED)  [![website](https://img.shields.io/badge/Website-MINED-orange?logo=homepage)](https://mined-lmm.github.io/) [![Slides](https://img.shields.io/badge/%F0%9F%93%8A%20Slides-MINED-BF55EC)](https://mined-lmm.github.io/MINED/MINED.pdf)




</h5>



## Table of Contents

- [Table of Contents](#table-of-contents)
- [🤗MINED](#mined)
- [🎯Main Results](#main-results)
- [🛠️Requirements and Installation](#️requirements-and-installation)
- [💥Inference](#inference)
- [🤖Evaluation](#evaluation)
- [🤝 Acknowledgments](#-acknowledgments)
- [📝 Citation](#-citation)



## 🤗MINED

Large Multimodal Models (LMMs) encode rich factual knowledge via cross-modal pre-training, yet their static representations struggle to maintain an accurate understanding of time-sensitive factual knowledge. Existing benchmarks remain constrained by static designs, inadequately evaluating LMMs' ability to understand time-sensitive knowledge. To address this gap, we propose <span style="font-weight: bold; color: #2E7D32;">MINED</span>, a comprehensive benchmark that evaluates temporal awareness along <b>6</b> key dimensions and <b>11</b> challenging tasks: <b>cognition, awareness, trustworthiness, understanding, reasoning, and robustness</b>. MINED is constructed from Wikipedia by two professional annotators, containing <b>2,104</b> time-sensitive knowledge samples spanning six knowledge types. Evaluating 15 widely used LMMs on MINED shows that Gemini-2.5-Pro achieves the highest average CEM score of 63.07, while most open-source LMMs still lack time understanding ability. Meanwhile, LMMs perform best on organization knowledge, whereas their performance is weakest on sport. To address these challenges, we investigate the feasibility of updating time-sensitive knowledge in LMMs through knowledge editing methods and observe that LMMs can effectively update knowledge via knowledge editing methods in single editing scenarios.

<div align="center">   <img src="figs\overview.png" width="700px"> </div>

You can download data 🤗 [Huggingface Dataset](https://huggingface.co/datasets/kailinjiang/MINED). And the expected structure of files is:


```text
MINED
|-- 
inference_data (json/jsonl)
|   |-- Dimension1_time_agnostic.json
|   |-- Dimension1_temporal_interval.json
|   |-- Dimension1_time_agnostic.json
|   |-- Dimension2_awareness_future.json
|   |-- Dimension2_awareness_past.json
|   |-- Dimension3_future_unanswerable_date.json
|   |-- Dimension3_previous_unanswerable_date.json
|   |-- Dimension4_understanding.json
|   |-- Dimension5_calculation.json
|   |-- Dimension5_ranking.json
|   |-- Dimension6_robustness.json
|-- imgs
|   |-- MINED_Image.zip
```

## 🎯Main Results

<div align="center">   <img src="figs\results.png" width="700px"> </div>


## 🛠️Requirements and Installation

```text
You can refer to https://github.com/open-compass/VLMEvalKit.git
```

<div align="center">   <img src="figs\install.png" width="700px"> </div>




## 💥Inference

```shell
python inference.py \
    --meta_save_path ./path/output \
    --model_name {base_model_name} \
    --data_eval_type {data_eval_type} \
    --max_new_token 10 \
    --image_path_prefix ./path/image_data
```

model_name refers to the model name defined in the VLMEvalKit\vlmeval\config.py file.

data_eval_type in ["time_agnostic", "timestamp", "temporal_interval", "awareness_future", "awareness_past", "future_unanswerable_date", "previous_unanswerable_date", "ranking", "understanding", "calculation", "robustness"]


## 🤖Evaluation

Evaluate **MINED**
```shell
python eval_code\cem_f1.py
```


## 🤝 Acknowledgments
We thank the following open-source projects for making this work possible:
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit.git) for the evaluation.


## 📝 Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :)

```bibtex
@article{jiang2025mined,
  title = {MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models},
  author={Jiang, Kailin and Jiang, Ning and Du, Yuntao and Ren, Yuchen and Li, Yuchen and Gao, Yifan and Bi, Jinhe and Ma, Yunpu and Liu, Qingqing and Wang, Xianhao and Jia, Yifan and Jiang, Hongbo and Hu, Yaocong and Li, Bin and Liu, Lei},
  year = {2025}
  url = {https://arxiv.org/pdf/2510.19457}
}
```



