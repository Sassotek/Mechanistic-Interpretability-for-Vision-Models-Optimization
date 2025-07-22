# Mechanistic-Interpretability-for-Vision-Models-Optimization
<!-- Badges -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NNMyHI6ySeZPHcacPNtQd6y8-yUvGMZX#scrollTo=6jzzOI7xEby3)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Camilla Giuliani on GitHub](https://img.shields.io/badge/Camilla–Giuliani–GitHub-181717?style=plastic&logo=github)](https://github.com/camygiuliani)
[![Pietro D'Annibale on GitHub](https://img.shields.io/badge/Pietro–D%E2%80%99Annibale–GitHub-181717?style=plastic&logo=github)](https://github.com/Sassotek)
<!--........-->

Project developed for the 2025 Computer Vision course, by Pietro D'Annibale (ID: 1917211) and Camilla Giuliani (ID: 1883207).


## Table of Contents 

1. [Colab notebook](#colab-notebook)
2. [Mechanistic Interpretability](#mechanistic-interpretability)
3. [Dataset](#dataset)  
4. [Model](#model)   
5. [Training](#training)  
6. [ACDC & pruning](#acdc-&-pruning)
7. [Considerations & final results](#considerations-&-final-results)    
8. [Future works](#future-works)
9. [References](#references)  
10. [License](#license)

---

## Colab notebook
Our notebook can be accessed via the following link:  
https://colab.research.google.com/drive/1NNMyHI6ySeZPHcacPNtQd6y8-yUvGMZX


## Mechanistic Interpretabilty
Mechanistic Interpretability is a growing research area that aims to reverse-engineer neural networks by understanding their internal components and computations. While it has been mainly applied to small language models, recent studies have started exploring its use in Vision Transformers as well. In this project, starting from a baseline ViT, we analyzed the model's internal behavior using one Mechanistic Interpretability technique. Our goal was to find a good trade-off between inference time and model accuracy by pruning the computational graph. Specifically, we identified and removed edges that were less significant for the final prediction, aiming to reduce computational complexity while preserving core model functionality.

## Dataset
The Tiny-ImageNet was downloaded from this Kaggle repository, cleaned and prepared for image classification tasks:    
 https://www.kaggle.com/datasets/wissamsalam/tiny-imagenet-cleaned-for-classification.  
This dataset is composed by 200 classes, with 64x64 images. For each class there are : 420 training samples, 50 validation samples and 80 test samples. To improve model robustness, we applied the following augmentations: random horizontal flip, random resized crop, random rotation, gaussian noise, random erasing, normalization with ImageNet mean & std, CutMix and MixUp.

## Model
To obtain the best baseline model possible,  several enanchments to the train were tried, such as: knowledge distillation, both with fine-tuned model and a clone baseline model. However no improvements were obtained using this. The use of AMP scaler and CosineAnnealing scheduler was a good idea to make the model learning better.

## Training
For the last version of our model was done a training cycle that produced a validation accuracy of ~35%.
 The train parameters were: 
- 30 epochs
- Batch size: 128
- Gradient flow visualization
- AMP: Automatic Mixed Precision
- CosineAnnealingLR 

Training was guided by (Soft Target) Cross Entropy Loss, and performance was evaluated using Accuracy.


## ACDC & pruning
We implemented the ACDC algorithm based on the method proposed in [1](Towards Automated Circuit Discovery for Mechanistic Interpretability,A. Conmy et al. ,2023), following its procedure to isolate the most relevant computational subcircuits in the Vision Transformer. KL divergence was used as the main metric to evaluate the impact of each component on the model’s output distribution. By progressively removing branches from the computational graph, we were able to prune the least relevant paths while preserving the core mechanisms responsible for the final prediction.

After experimenting with different values of the pruning threshold τ, the final version of our model retained only the most essential connections. Specifically, 81 out of 518 edges were pruned from the computational graph in the selected configuration.

## Considerations & final results
The final results show that the initial loss in accuracy derived from the pruning is recovered through a training cycle. Although, the reduction of inference time was not satisfactory consisting in a a difference of ~2 sec from the pruned model and the original one.

| Model                          |   Test Loss | Test Accuracy   | InferenceTime   |
|--------------------------------|-------------|-----------------|-----------------|
| Baseline Model                 |     361.099 | 36.25%          | 61.31 s         |
| Baseline Model Pruned          |     383.528 | 33.00%          | 59.87 s         |
| Upgraded Baseline Model        |     368.933 | 36.34%          | 60.51 s         |
| Upgraded Baseline Model Pruned |     365.003 | 36.07%          | 58.11 s         |

## Future works
Optimizing the balance between inference efficiency and accuracy remains an open challenge. Future directions could include experimenting with higher τ values while compensating for accuracy loss using techniques like Knowledge Distillation or selective re-training of key components.

We also looked into Edge Attribution Patching (EAP), a recent method proposed in [2](Attribution Patching Outperforms Automated Circuit Discovery,A. Syed et al,2024), as a faster and more efficient alternative to ACDC. It estimates the importance of each edge using attribution scores from a single backward pass. Although the idea is promising and has shown good results in other contexts, we decided not to implement it in this work due to limited gains in our specific setup. Still, combining EAP with ACDC—as suggested in [2]-might be worth exploring in future work.

## References 
- [1][A. Conmy et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. In: Advances
in Neural Information Processing Systems 36 (NeurIPS 2023) ](https://arxiv.org/abs/2304.14997)
- [2][A. Syed, C. Rager and A.Conmy, (2024). Attribution Patching Outperforms Automated Circuit Discovery,
BlackboxNLP 2024](https://arxiv.org/abs/2310.10348) 
- [3][A. Vaswani et al. (2017). Attention is all you need.In: Advances in Neural Information Processing Systems  36 (NeurIPS 2017) ](https://arxiv.org/abs/1706.03762)
- [4] [TinyImageNet dataset](https://www.kaggle.com/datasets/wissamsalam/tiny-imagenet-cleaned-for-classification)
- [5][A.Dosovitskiy et al.(2021).An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
-[6][VISO.ai](https://viso.ai/deep-learning/vision-transformer-vit/)
- [7][Einops Guide](https://nbviewer.org/github/arogozhnikov/einops/blob/main/docs/1-einops-basics.ipynb)

## License
This project is released under the MIT License.