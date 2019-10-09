# Introduction
This Repo is used to Collect Few-shot Learning Area Papers, welcome to supplement in the Issues
## 目录
<!-- vim-markdown-toc GitLab -->

* [UnReading](#unreading)
* [Image Classification](#image-classification)
  * [Optimize Based Few-shot Learning](#optimize-based-few-shot-learning)
  * [Generative Based Few-shot Learning](#generative-based-few-shot-learning)
    * [Summary](#summary)
    * [Papers](#papers)
  * [Metric Based Few-shot Learning](#metric-based-few-shot-learning)
    * [Summary](#summary-1)
    * [Traditional](#traditional)
      * [Semi-Supervised](#semi-supervised)
      * [Supervised](#supervised)
  * [Special (such as Architecture?)](#special-such-as-architecture)
    * [External Memory](#external-memory)
    * [Architecture](#architecture)
    * [Task Representation and Measure](#task-representation-and-measure)
    * [Multi Label Image Classification](#multi-label-image-classification)
    * [Add Additional Informations](#add-additional-informations)
  * [Results in Datasets](#results-in-datasets)
    * [Omniglot](#omniglot)
    * [mini-Imagenet](#mini-imagenet)
    * [tiredImagenet](#tiredimagenet)
    * [Imagenet](#imagenet)
    * [CUB 2011](#cub-2011)
* [Object Detection](#object-detection)
* [Segementation](#segementation)
* [Generative Model](#generative-model)
* [Domain Adaptation](#domain-adaptation)
* [Reinforcement Learning](#reinforcement-learning)
* [Others](#others)

<!-- vim-markdown-toc -->

## UnReading
- [ ] [CVPR 2019] Generating Classification Weights with GNN Denoising Autoencoders for Few-Shot Learning
- [ ] [CVPR 2019] Large-Scale Few-Shot Learning- Knowledge Transfer With Class Hierarchy
- [ ] [CVPR 2019] Spot and Learn- A Maximum-Entropy Patch Sampler for Few-Shot Image Classification
- [ ] [CVPR 2017] Re-ranking Person Re-identification with k-reciprocal Encoding
- [ ] [CVPR 2019] Meta-Transfer Learning for Few-Shot Learning
- [ ] [CVPR 2019] Image Deformation Meta-Networks for One-Shot Learning
- [ ] [CVPR 2019] Generalized Zero- and Few-Shot Learning via Aligned Variational Autoencoders
- [ ] [CVPR 2019] Adaptively Connected Neural Networks
- [ ] [ICML 2019] Infinite Mixture Prototypes for Few-shot Learning
- [ ] [ICCV 2019] Deep Meta Learning for Real-Time Target-Aware Visual Tracking
- [ ] [ICCV 2019] Few-Shot Learning with Global Class Representations
- [ ] [ICCV 2019] MetaPruning-Meta Learning for Automatic Neural Network Channel Pruning


## Image Classification
### Optimize Based Few-shot Learning
- [x] [ICLR 2017] Optimization as a Model for Few-shot Learning
- [x] [CVPR 2019] Dense Classification and Implanting for Few-Shot Learning
- [x] [ICML 2019] LGM-Net: Learning to Generate Matching Networks for Few shot Learning
- [x] [CVPR 2018]Dynamic Few-Shot Visual Learning without Forgetting

### Generative Based Few-shot Learning
#### Summary
- [x] [NIPS 2018 Bengio] MetaGAN An Adversarial Approach to Few-Shot Learning

#### Papers
- [x] [ICCV 2017] Low-shot Visual Recognition by Shrinking and Hallucinating Features
- [x] [CVPR 2018] Low-Shot Learning from Imaginary Data
- [x] [NIPS 2018] Low-shot Learning via Covariance-Preserving Adversarial Augmentation Networks

### Metric Based Few-shot Learning
#### Summary
- [x] [ICLR 2019] A CLOSER LOOK AT FEW-SHOT CLASSIFICATION

#### Traditional
- [x] [ICML 2015] Siamese Neural Networks for One-Shot Image Recognition
- [x] [NIPS 2016] Matching Networks for One Shot Learning
- [x] [NIPS 2017] Prototypical Networks for Few-shot Learning
- [x] [CVPR 2018] Learning to Compare：Relation Network for Few-Shot Learning


##### Semi-Supervised
- [x] [ICLR 2019] Learning to Propagate Labels-transductive Propagation Network for Few-shot Learning
- [x] [CVPR 2019] Edge-Labeling Graph Neural Network for Few-shot Learning
- [x] [ICLR 2018] Meta-Learning for Semi-Supervised Few-Shot Classification

##### Supervised
- [x] [NIPS 2018] TADAM-Task dependent adaptive metric for improved few-shot learning
- [x] [CVPR 2019] Finding Task-Relevant Features for Few-Shot Learning by Category Traversal

- [x] [ICML 2019] TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning
- [x] [CVPR 2019] Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning


### Special (such as Architecture?)
#### External Memory
- [x] [ICLR 2019] Adaptive Posterior Learning-Few-Shot Learning with a Surprise-Based Memory Module
- [x] [CVPR 2018] Memory Matching Networks for One-Shot Image Recognition

#### Architecture
- [x] [ICML 2017] Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

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

## Domain Adaptation
- [x] [NIPS 2017] Few-Shot Adversarial Domain Adaptation
- [x] [ICCV 2019] Bidirectional One-Shot Unsupervised Domain Mapping

## Reinforcement Learning
- [x] [ICML 2019] Few-Shot Intent Inference via Meta-Inverse Reinforcement Learning

## Others
- [x] [IJCAI 2019] Incremental Few-Shot Learning for Pedestrian Attribute Recognition
