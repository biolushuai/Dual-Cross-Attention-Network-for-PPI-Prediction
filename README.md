This repository is the implementation of our Paper under review.


# DCAPPI: PPI Prediction Through Dual Cross-Attention Network


## Abstract
Characterizing protein-protein interactions (PPI) is essential for deciphering fundamental biological mechanisms. However, experimental PPI determination remains time-consuming and expensive, driving the adoption of deep learning as an efficient and accurate computational approach. Current deep learning-based PPI prediction models typically process both intra- and inter-protein as isolated units in feature extraction, thereby ignoring mutual information transfer within a single protein and the interacting pair. To address this limitation, we propose DCAPPI (Dual Cross-Attention network for Protein-Protein Interaction prediction), a novel framework leveraging dual cross-attention modules for hierarchical feature fusion at both intra- and inter-protein levels. First, the Channel Cross-Attention (CCA) module processes protein sequence and structure as distinct input channels. It generates deep intra-protein representations by performing cross-attention between sequence-derived and structure-derived tokens, achieving multimodal feature integration. Second, the Partner Cross-Attention (PCA) module models the target protein and its interacting partner as a pair of correlative units. By performing cross-attention operations across these units, it enables collaborative feature fusion and constructs context-aware inter-protein interaction features. Evaluation results indicate that DCAPPI achieves superior performance over the state-of-the-art methods on benchmark datasets. 

## Model Architecture

DCAPPI employs a hierarchical dual cross-attention framework:

- **Intra-protein level**: The Channel Cross-Attention (CCA) module fuses multimodal features by applying bidirectional cross-attention between GAT-based structural embeddings and GRU-based sequential embeddings derived from protein language models (ESM-2, ProtBert, or ProtT5).
- **Inter-protein level**: The Partner Cross-Attention (PCA) module captures interaction patterns by computing cross-attention between the two interacting proteins, followed by a softmax-weighted aggregation to produce compact, context-aware pair representations.
- **Multi-task supervision**: Beyond PPI prediction, auxiliary tasks (GO term prediction and subcellular localization prediction) are jointly optimized via an automatically weighted multi-task loss, providing additional biological supervision.

## Dependencies

- python == 3.7+
- pytorch == 1.7.1+
- torch-geometric (PyG) == 2.0.4
- torch-cluster == 1.5.9
- torch-scatter == 2.0.5
- torch-sparse == 0.6.8
- torch-spline-conv == 1.2.0
- scikit-learn == 1.0.2
- scipy == 1.7.3
- numpy == 1.21.5
- pandas == 1.3.5
- tqdm
- 
## Datasets

We evaluate DCAPPI on four benchmark datasets:

| Dataset | Task | Species | Description |
|---------|------|---------|-------------|
| Yeast | Binary PPI | *S. cerevisiae* | Protein-protein interactions in yeast |
| Multi-species | Binary PPI | Cross-species | PPI data with varying sequence identity thresholds (any/01/10/25/40) |
| Multi-class | 7-class PPI | — | Fine-grained interaction type prediction (7 interaction categories) |
| Gold Standard | Binary PPI | *H. sapiens* | Data-leakage-free benchmark from Bernett et al. (2024) [[1]](#references) |

### Data Preparation

The preprocessed data for Yeast, Multi-species, and Multi-class datasets (including contact maps, protein language model embeddings, GO term labels, and subcellular localization labels) can be downloaded from the following link:
[Download Link](https://pan.baidu.com/s/1YQoNeZ8zDU_4_BiC_JGfkA?pwd=hdzd) (password: hdzd)

## Usage

### 1. Binary PPI Prediction (Yeast / Multi-species)

Train and evaluate on binary PPI tasks:

```bash
# Yeast dataset
python main.py --dataset yeast --feat_dim 1280 --heads 4 --batch_size 16 --device 0

# Multi-species dataset (with sequence identity threshold)
python main.py --dataset multi_species --identity 01 --feat_dim 1280 --heads 4 --batch_size 16 --device 0
```

### 2. Multi-class PPI Prediction

Train and evaluate on the 7-class interaction type prediction task:

```bash
python main.py --dataset multi_class --feat_dim 1280 --heads 4 --batch_size 16 --device 0
```

### 3. 5-Fold Cross-Validation (Yeast)

For robust evaluation on the yeast dataset:

```bash
python main_5_folds.py --dataset yeast --distance 8.0 --lm esm-2 --feat_dim 1280 --heads 4 --batch_size 16 --device 0
```

Key parameters for 5-fold CV:

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--dataset` | Dataset name | `yeast` |
| `--distance` | Contact map distance threshold | `4.0`, `6.0`, `8.0`, `10.0` |
| `--lm` | Protein language model | `esm-2`, `ProtBert-BFD`, `ProtT5-XL-UniRef50` |
| `--feat_dim` | Feature dimension | `1280` (ESM-2), `1024` (ProtBert/ProtT5) |
| `--heads` | Attention heads | `4` |
| `--batch_size` | Batch size | `16` |
| `--device` | CUDA device ID | `0`, `1`, etc. |

## Training Details

- **Optimizer**: Adam with learning rate = 0.001
- **Loss Function**: Binary cross-entropy (PPI) + binary cross-entropy (GO) + binary cross-entropy (localization), balanced via Automatic Weighted Loss
- **Epochs**: 200 (binary tasks) / 80 (5-fold CV)
- **Batch Size**: 16
- **Metrics**: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-score, MCC, AUC, AUPR
## Baseline Methods

We compare DCAPPI against the following state-of-the-art PPI prediction methods:

| Method | Repository | Description |
|--------|-----------|-------------|
| TAGPPI | [GitHub](https://github.com/xzenglab/TAGPPI) | Topology-aware graph neural network for PPI |
| PIPR | [GitHub](https://github.com/muhaochen/seq_ppi) | Protein-protein interaction prediction based on Siamese residual RCNN |
| Struct2Graph | [GitHub](https://github.com/baranwa2/Struct2Graph) | Graph neural network for predicting protein-protein interactions |
| DPPI | [GitHub](https://github.com/hashemifar/DPPI/) | Deep learning architecture for PPI prediction |

## References

[1] Bernett, J., Blumenthal, D. B., & List, M. (2024). Cracking the black box of deep sequence-based protein-protein interaction prediction. *Briefings in Bioinformatics*, 25(2), bbae076. [DOI: 10.1093/bib/bbae076](https://doi.org/10.1093/bib/bbae076)

## Acknowledgements

We gratefully acknowledge the open-source contributions from:

- [CollaPPI](https://github.com/Wenjian-Ma/CollaPPI) for the data preprocessing pipeline and multi-task learning framework
- [PPIT-BAN](https://github.com/NWPU-903PR/PPIT-BAN) for the bilinear attention network design inspiration
- [ESM](https://github.com/facebookresearch/esm) for protein language model embeddings
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) for graph neural network implementations
