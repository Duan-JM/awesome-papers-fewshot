# Introduction
This Repo is used to Collect Few-shot Learning Area Papers, welcome to supplement in the Issues
## 目录
<!-- vim-markdown-toc GitLab -->

- [Introduction](#introduction)
  - [目录](#%e7%9b%ae%e5%bd%95)
  - [ToDo](#todo)
  - [ReadLater](#readlater)
    - [Generative](#generative)
    - [Optimize](#optimize)
  - [Focus Reading](#focus-reading)
  - [Sorting](#sorting)
  - [Image Classification](#image-classification)
  - [Summary](#summary)
    - [Optimize Based Few-shot Learning](#optimize-based-few-shot-learning)
    - [Generative Based Few-shot Learning](#generative-based-few-shot-learning)
      - [Summary](#summary-1)
      - [Papers](#papers)
    - [Metric Based Few-shot Learning](#metric-based-few-shot-learning)
      - [Traditional](#traditional)
        - [Semi-Supervised](#semi-supervised)
        - [Supervised](#supervised)
    - [Special (such as Architecture?)](#special-such-as-architecture)
      - [External Memory](#external-memory)
      - [Architecture](#architecture)
      - [Task Representation and Measure](#task-representation-and-measure)
      - [Multi Label Image Classification](#multi-label-image-classification)
      - [Add Additional Informations](#add-additional-informations)
      - [Self-training](#self-training)
    - [Results in Datasets](#results-in-datasets)
      - [Omniglot](#omniglot)
      - [mini-Imagenet](#mini-imagenet)
      - [tiredImagenet](#tiredimagenet)
      - [Imagenet](#imagenet)
      - [CUB 2011](#cub-2011)
  - [Object Detection](#object-detection)
  - [Segementation](#segementation)
  - [Generative Model](#generative-model)
  - [Domain Adaptation](#domain-adaptation)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Visual Tracking](#visual-tracking)
  - [Others](#others)

<!-- vim-markdown-toc -->
## ToDo
- [ ] (WIP)Add one comment to each works
- [ ] (WIP)Add results tables for all works
- [ ] Add paper link and opensource code for each works

## ReadLater
- [ ] [NIPS 2018]  ([paper](https://arxiv.org/pdf/1805.08113.pdf))Stacked Semantics-Guided Attention Model for Fine-Grained Zero-Shot Learning
- [ ] [NIPS 2018] ([paper](https://papers.nips.cc/paper/7471-generalized-zero-shot-learning-with-deep-calibration-network.pdf)) Generalized Zero-Shot Learning with Deep Calibration Network
- [ ] [ICCV 2019] AMP- Adaptive Masked Proxies for Few-Shot Segmentation
- [ ] [CVPR 2019] ([paper](https://arxiv.org/pdf/1902.11123.pdf) [code](https://github.com/MSiam/AdaptiveMaskedProxies.)) SAR Image Classification Using Few-shot Cross-domain Transfer Learning
- [ ] [NIPS 2019] ([paper](https://arxiv.org/abs/1910.07677)) Cross Attention Network for Few-shot Classification
- [ ] [NIPS 2019] ([code](https://github.com/apple2373/MetaIRNet)) Metal-Reinforced Synthetic Data for One-Shot Fine-Grained Visual Recognition

### Generative
- [ ] [CVPR 2017] ([paper](https://arxiv.org/pdf/1612.02559.pdf) [code](https://github.com/rkwitt/GuidedAugmentation)) AGA - Attribute-Guided Augmentation

### Optimize
- [ ] [NIPS 2019] ([paper](https://arxiv.org/pdf/1810.07218.pdf) [code](https://github.com/renmengye/inc-few-shot-attractor-public))Incremental Few-Shot Learning with Attention Attractor Networks.pdf

## Focus Reading
- [x] [arXiv 2019] ([paper](https://arxiv.org/pdf/1902.10482.pdf)) Few-Shot Text Classification with Induction Network

    Introduce dynamic routing to generate better class representations. One real industrial project.

- [ ] [arXiv 2019] ([paper](https://arxiv.org/pdf/1902.08605) [code](https://github.com/gabrielhuang/centroid-networks)) Centroid Networks for Few-Shot Clustering and Unsupervised Few-Shot Classification

- [x] [ICCV 2019] ([paper](https://arxiv.org/pdf/1908.05257)) Few-Shot Learning with Global Class Representations 
    * Synthesis new samples to elleviate the data imbalance problem between Base and Novel Classes.
    * During training, compute two losses, one is the original losses, the other is the score for the whole classes including noval classes.

- [x] [arXiv 2019]  (RECOMMENDED!) ([paper](https://arxiv.org/pdf/1902.03545.pdf)) TASK2VEC- Task Embedding for Meta-Learning
- [x] [ICML 2019] (RECOMMENDED!) ([paper](https://arxiv.org/pdf/1902.04552.pdf)) Infinite Mixture Prototypes for Few-shot Learning

    * Point out that data distribution for one class are not uni-model (Verify in my experiments too).
    * (Clustering methods) Semi-Supervised methods for prototypical networks. Show this methods even suit for unsupervised situations(protentially).
    * Improve on Alphabets dataset, remain or improve on omniglot and mini-imagenet.

- [x] [AAAI 2019] ([paper](https://www.researchgate.net/publication/335296764_Attention-Based_Multi-Context_Guiding_for_Few-Shot_Semantic_Segmentation)) Attention-based Multi-Context Guiding for Few-Shot Semantic Segmentation

    Utilize the output of the different layers between query branch and support branch to gain more context informations.

- [x] [ICLR 2018] ([paper](https://arxiv.org/pdf/1803.00676.pdf) [code](https://github.com/renmengye/few-shot-ssl-public)) Meta-Learning for Semi-Supervised Few-Shot Classification
    * Using soft K-means to refine the prototypes, then using varient ways(training methods) to eliminate the outline points.
    * Create new datasets - tiredImagenet

- [x] [IJCAI 2019] ([paper](https://arxiv.org/pdf/1905.04042) [code](https://github.com/liulu112601/PPN)) Prototype Propagation Networks (PPN) for Weakly-supervised Few-shot Learning on Category Graph

    Maually build an category graph, then add parents label's class represention into the child class representations.

## Sorting
- [ ] [CVPR 2018] ([paper](https://arxiv.org/pdf/1712.00981.pdf)) Feature Generating Networks for Zero-ShsoetenLearning
- [ ] [CVPR 2018] ([paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Temporal_Hallucinating_for_CVPR_2018_paper.pdf)) Temporal Hallucinating for Action Recognition with Few Still Images
- [ ] [ECCV 2018] ([paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Fang_Zhao_Dynamic_Conditional_Networks_ECCV_2018_paper.pdf)) Dynamic Conditional Networks for FewShot Learning
- [ ] [ICIP 2018] Discriminative Hallucination for Multi-Modal Few-Shot Learning
- [ ] [Nature 子刊 MI 2018] Continuous Learning of Context-dependent Processing in Neural Networks
- [ ] [CVPR 2019] Generating Classification Weights with GNN Denoising Autoencoders for Few-Shot Learning
- [ ] [CVPR 2019] Large-Scale Few-Shot Learning- Knowledge Transfer With Class Hierarchy
- [ ] [CVPR 2019] Spot and Learn- A Maximum-Entropy Patch Sampler for Few-Shot Image Classification
- [ ] [CVPR 2019] Meta-Transfer Learning for Few-Shot Learning
- [ ] [CVPR 2019] Image Deformation Meta-Networks for One-Shot Learning
- [ ] [CVPR 2019] Generalized Zero- and Few-Shot Learning via Aligned Variational Autoencoders
- [ ] [CVPR 2019] Baby steps towards few-shot learning with multiple semantics
- [ ] [CVPR 2019] Few-Shot Learning with Localization in Realistic Settings
- [ ] [CVPR 2019] Variational Prototyping-Encoder- One-Shot Learning with Prototypical Images
- [ ] [NIPS 2019] Adaptive Cross-Modal Few-shot Learning

## Image Classification
## Summary
- [x] [arXiv 2019] Generalizing from a Few Examples A Survey on Few-Shot Learning
- [x] [ICLR 2019] A Closer Look At Few-shot Classification

### Optimize Based Few-shot Learning
- [x] [CVPR 2017] Learning to Learn Image Classifiers with Visual Analogy
- [x] [ICLR 2017] Optimization as a Model for Few-shot Learning

- [x] [CVPR 2018] Dynamic Few-Shot Visual Learning without Forgetting
- [x] [CVPR 2018] Low-Shot Learning with Imprinted Weights

    Passing, generate weights for classifier. (I DONOT like it, the exp only compare to matching networks and "Generative + classifier")

- [x] [CVPR 2019] Dense Classification and Implanting for Few-Shot Learning
- [x] [ICML 2019] LGM-Net: Learning to Generate Matching Networks for Few shot Learning
- [x] [CVPR 2019] TAFE-Net- Task-Aware Feature Embeddings for Low Shot Learning

    Use a meta-learner to generate parameters for the feature extractor

### Generative Based Few-shot Learning
#### Summary
- [x] [NIPS 2018 Bengio] MetaGAN An Adversarial Approach to Few-Shot Learning

#### Papers
- [x] [ICCV 2017] Low-shot Visual Recognition by Shrinking and Hallucinating Features
- [x] [CVPR 2018] Low-Shot Learning from Imaginary Data
- [x] [NIPS 2018] Low-shot Learning via Covariance-Preserving Adversarial Augmentation Networks

### Metric Based Few-shot Learning
#### Traditional
- [x] [ICML 2012] One-Shot Learning with a Hierarchical Nonparametric Bayesian Model

    Using Hierarchical Bayesian Model after extracted features. Which is similar to build the category graph method in IJCAI 2019.

- [x] [ICML 2015] Siamese Neural Networks for One-Shot Image Recognition
- [x] [NIPS 2016]  (RECOMMENDED!) Matching Networks for One Shot Learning
- [x] [NIPS 2017]  (RECOMMENDED!) Prototypical Networks for Few-shot Learning
- [x] [CVPR 2018]  (RECOMMENDED!) Learning to Compare：Relation Network for Few-Shot Learning


##### Semi-Supervised
- [x] [ICLR 2018] Meta-Learning for Semi-Supervised Few-Shot Classification

- [x] [CVPR 2018] (RECOMMENDED!) Low-Shot Learning With Large-Scale Diffusion
- [x] [ICLR 2019] Learning to Propagate Labels-transductive Propagation Network for Few-shot Learning
- [x] [CVPR 2019] Edge-Labeling Graph Neural Network for Few-shot Learning

##### Supervised
- [x] [NIPS 2018] (RECOMMENDED!) TADAM-Task dependent adaptive metric for improved few-shot learning 
- [x] [CVPR 2019] Finding Task-Relevant Features for Few-Shot Learning by Category Traversal

- [x] [ICML 2019]  (RECOMMENDED!) TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning
- [x] [CVPR 2019] Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning


### Special (such as Architecture?)
#### External Memory
- [x] [ICML 2016] Meta-Learning with Memory-Augmented Neural Networks

    This work lead NTM into the image classification, technically, this work should not belong to the few-shot problems.
    This method can identify the image labels, even the true label of current image are inputed along with the next image.

- [x] [ICLR 2019] Adaptive Posterior Learning-Few-Shot Learning with a Surprise-Based Memory Module
- [x] [CVPR 2018] Memory Matching Networks for One-Shot Image Recognition

#### Architecture
- [x] [ICML 2017] Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
- [x] [CVPR 2019] Task-Agnostic Meta-Learning for Few-shot Learning

    A training method force model to learn a unbiased initial model without over-performing on some particular tasks.

#### Task Representation and Measure
- [x] [ICCV 2019] TASK2VEC- Task Embedding for Meta-Learning

#### Multi Label Image Classification
- [x] [CVPR 2019 oral] LaSO-Label-Set Operations networks for multi-label few-shot learning

#### Add Additional Informations
- [x] [ICCV 2019] Learning Compositional Representations for Few-Shot Recognition

    Add additional annotations to the classes.

- [x] [CVPR 2019] Few-shot Learning via Saliency-guided Hallucination of Samples

    Form segementations and mix up, aiming at eliminates the back ground noise.

- [x] [ICCV 2019] Boosting Few-Shot Visual Learning with Self-Supervision

    Self-supervision means to rotate itself, and compute two losses.

#### Self-training

- [x] [NIPS 2019] Learning to Self-Train for Semi-Supervised Few-Shot Classification.pdf


### Results in Datasets
#### [Omniglot](https://github.com/brendenlake/omniglot)
#### [mini-Imagenet](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning)
#### [tiredImagenet](https://arxiv.org/abs/1803.00676)
#### [Imagenet](http://image-net.org)
#### [CUB 2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

## Object Detection
- [x] [CVPR 2019] RepMet-Representative-based Metric Learning for Classification and Few-shot Object Detection
- [x] [CVPR 2019] Few-shot Adaptive Faster R-CNN
- [x] [CVPR 2019] Feature Selective Anchor-Free Module for Single-Shot Object Detection
- [x] [ICCV 2019] Few-shot Object Detection via Feature Reweighting

## Segementation
- [x] [CVPR 2019] CANet- Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning

## Generative Model
- [x] [ICCV 2019] Few-Shot Unsupervised Image-to-Image Translation
- [x] [CVPR 2018] Multi-Content GAN for Few-Shot Font Style Transfer

## Domain Adaptation
- [x] [NIPS 2017] Few-Shot Adversarial Domain Adaptation
- [x] [ICCV 2019] Bidirectional One-Shot Unsupervised Domain Mapping

## Reinforcement Learning
- [x] [ICML 2019] Few-Shot Intent Inference via Meta-Inverse Reinforcement Learning

## Visual Tracking
- [x] [ICCV 2019] Deep Meta Learning for Real-Time Target-Aware Visual Tracking

## Others
- [x] [IJCAI 2019] Incremental Few-Shot Learning for Pedestrian Attribute Recognition
- [x] [AAAI 2018] AffinityNet- Semi-supervised Few-shot Learning for Disease Type Prediction

    Use few-shot method to enhance oringal disease type prediction
