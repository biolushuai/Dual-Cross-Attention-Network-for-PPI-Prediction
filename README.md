This repository is the implementation of our Paper under review.


# DCAPPI: PPI Prediction Through Dual Cross-Attention Network


# Abstract
Characterizing protein-protein interactions (PPI) is essential for deciphering fundamental biological mechanisms. However, experimental PPI determination remains time-consuming and expensive, driving the adoption of deep learning as an efficient and accurate computational approach. Current deep learning-based PPI prediction models typically process both intra- and inter-protein as isolated units in feature extraction, thereby ignoring mutual information transfer within a single protein and the interacting pair. To address this limitation, we propose DCAPPI (Dual Cross-Attention network for Protein-Protein Interaction prediction), a novel framework leveraging dual cross-attention modules for hierarchical feature fusion at both intra- and inter-protein levels. First, the Channel Cross-Attention (CCA) module processes protein sequence and structure as distinct input channels. It generates deep intra-protein representations by performing cross-attention between sequence-derived and structure-derived tokens, achieving multimodal feature integration. Second, the Partner Cross-Attention (PCA) module models the target protein and its interacting partner as a pair of correlative units. By performing cross-attention operations across these units, it enables collaborative feature fusion and constructs context-aware inter-protein interaction features. Evaluation results indicate that DCAPPI achieves superior performance over the state-of-the-art methods on benchmark datasets. 



Data preparation
---
1. The relevant data and trained model of _yeast_ (~6.14G) can be available at the [Link](https://pan.baidu.com/s/1kknFC2gpayvxLM_1sqwO7w?pwd=1234) and [Link](https://pan.baidu.com/s/10235SAt1oknq8TmtOosQiA?pwd=1234).

2. The relevant data and trained model of _multi-species_ (~24.16G) can be available at the [Link](https://pan.baidu.com/s/1kQHXCAQxzNO5peLqJni8xg?pwd=1234).

3. The relevant data and trained model of _multi-class_ (~11.60G) can be available at the [Link](https://pan.baidu.com/s/18VNZJzcRQCN8myJ8Pb6SAA?pwd=1234).

4. Unzip the above file to the corresponding directory (e.g., dictionary_yeast.tar.gz should be extracted to `./data/yeast`).

5. If you want to train or test the model on different datasets, please modify the parameter settings in the code.

Train
---
`python main.py`

Test
---
`python test.py` used to reproduct the performence recorded in the paper.

`python test_mul.py` multiplication for mutual interaction. Tranined model can be available at the [Link](https://pan.baidu.com/s/1QgK3w80w08U_Ywl3aBwc3w?pwd=1234), which is evaluated on yeast dataset.

`python test_tran_mul.py` Transposed multiplication for mutual interaction. Trained model can be available at the [Link](https://pan.baidu.com/s/1E_t8KWFyZfQvo9qCxb25Ag?pwd=1234), which is evaluated on yeast dataset.

Baseline
---

TAGPPI: [https://github.com/xzenglab/TAGPPI](https://github.com/xzenglab/TAGPPI).

PIPR: [https://github.com/muhaochen/seq_ppi](https://github.com/muhaochen/seq_ppi).

Struct2Graph: [https://github.com/baranwa2/Struct2Graph](https://github.com/baranwa2/Struct2Graph).

DPPI: [https://github.com/hashemifar/DPPI/](https://github.com/hashemifar/DPPI/).


We retrained the baseline method, Struct2Graph, with its default parameters using the yeast dataset in our paper.

1. The preprocessed data in yeast dataset used for Struct2Graph can be available at the [Link](https://pan.baidu.com/s/1mrJ5HQ2wMp1Wv0D3YI72Cg?pwd=1234).

2. The trained model on yeast dataset can be available at the [Link](https://pan.baidu.com/s/19KpAuXthWU6RZTF5FORPhA?pwd=1234), which is used to reproduce the performance of Struct2Graph recorded in our paper.

    Under the path of Struct2Graph-master/ :  `python test.py`

3. If you want to retrain the Struct2Graph on the yeast dataset in our paper:

    Under the path of Struct2Graph-master/ :  `python k-fold-CV.py`

Cite our work
---
if you use the conclusion, code, or data in our work, please cite:
```
@ARTICLE{10465250,
  author={Ma, Wenjian and Bi, Xiangpeng and Jiang, Huasen and Zhang, Shugang and Wei, Zhiqiang},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={CollaPPI: A Collaborative Learning Framework for Predicting Protein-Protein Interactions}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Proteins;Collaboration;Task analysis;Feature extraction;Protein engineering;Deep learning;Vectors;protein-protein interaction;multi-task learning;graph neural network;protein representation learning},
  doi={10.1109/JBHI.2024.3375621}}
```
