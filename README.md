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

The preprocessed data for Yeast, Multi-species, and Multi-class datasets (including contact maps, protein language model embeddings, GO term labels, and subcellular localization labels) can be downloaded from the following links:
[Download Link](https://pan.baidu.com/s/1YQoNeZ8zDU_4_BiC_JGfkA?pwd=hdzd) (password: hdzd)

