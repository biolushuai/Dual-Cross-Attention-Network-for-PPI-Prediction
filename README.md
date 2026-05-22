This repository is the implementation of our Paper under review.


# DCAPPI: PPI Prediction Through Dual Cross-Attention Network


# Abstract
Characterizing protein-protein interactions (PPI) is essential for deciphering fundamental biological mechanisms. However, experimental PPI determination remains time-consuming and expensive, driving the adoption of deep learning as an efficient and accurate computational approach. Current deep learning-based PPI prediction models typically process both intra- and inter-protein as isolated units in feature extraction, thereby ignoring mutual information transfer within a single protein and the interacting pair. To address this limitation, we propose DCAPPI (Dual Cross-Attention network for Protein-Protein Interaction prediction), a novel framework leveraging dual cross-attention modules for hierarchical feature fusion at both intra- and inter-protein levels. First, the Channel Cross-Attention (CCA) module processes protein sequence and structure as distinct input channels. It generates deep intra-protein representations by performing cross-attention between sequence-derived and structure-derived tokens, achieving multimodal feature integration. Second, the Partner Cross-Attention (PCA) module models the target protein and its interacting partner as a pair of correlative units. By performing cross-attention operations across these units, it enables collaborative feature fusion and constructs context-aware inter-protein interaction features. Evaluation results indicate that DCAPPI achieves superior performance over the state-of-the-art methods on benchmark datasets. 


Baseline
---

TAGPPI: [https://github.com/xzenglab/TAGPPI](https://github.com/xzenglab/TAGPPI).

PIPR: [https://github.com/muhaochen/seq_ppi](https://github.com/muhaochen/seq_ppi).

Struct2Graph: [https://github.com/baranwa2/Struct2Graph](https://github.com/baranwa2/Struct2Graph).

DPPI: [https://github.com/hashemifar/DPPI/](https://github.com/hashemifar/DPPI/).
