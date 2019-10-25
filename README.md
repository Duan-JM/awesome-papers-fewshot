## 目录
<!-- vim-markdown-toc GitLab -->

* [Introduction](#introduction)
* [ToDo](#todo)
* [ReadLater](#readlater)
* [Sorting](#sorting)
* [Image Classification](#image-classification)
  * [Summary](#summary)
  * [Optimize Based Few-shot Learning (Trying to generate Classifiers' parameters)](#optimize-based-few-shot-learning-trying-to-generate-classifiers-parameters)
  * [Generative Based Few-shot Learning](#generative-based-few-shot-learning)
    * [Summary](#summary-1)
    * [Papers](#papers)
  * [Metric Based Few-shot Learning](#metric-based-few-shot-learning)
    * [Traditional](#traditional)
      * [Semi-Supervised](#semi-supervised)
      * [Supervised](#supervised)
  * [Special (such as Architecture?)](#special-such-as-architecture)
    * [External Memory](#external-memory)
    * [Architecture](#architecture)
    * [Task Representation and Measure](#task-representation-and-measure)
    * [Multi Label Image Classification](#multi-label-image-classification)
    * [Add Additional Informations](#add-additional-informations)
    * [Self-training](#self-training)
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
* [Visual Tracking](#visual-tracking)
* [Others](#others)

<!-- vim-markdown-toc -->
## Introduction

This Repo is used to Collect Few-shot Learning Area Papers, welcome to supplement in the Issues.

## ToDo
- [ ] (WIP)Add one comment to each works
- [ ] (WIP)Add results tables for all works
- [ ] Add paper link and opensource code for each works


## ReadLater
- [ ] [NIPS 2018] Stacked Semantics-Guided Attention Model for Fine-Grained Zero-Shot Learning
- [ ] [NIPS 2018] Generalized Zero-Shot Learning with Deep Calibration Network
- [ ] [CVPR 2017] AGA - Attribute-Guided Augmentation



## Sorting
- [ ] [arXiv 2018 REPTILE] On First-Order Meta-Learning Algorithms
- [ ] [OpenReview? 2019 IDeMe-Net] Image deformation meta-networks for one-shot learning

- [ ] [arXiv 2019] Centroid Networks for Few-Shot Clustering and Unsupervised Few-Shot Classification
- [ ] [arXiv 2019] AMP-Adaptive Masked Proxies for Few-Shot Segmentation
- [ ] [CVPR 2018] Feature Generating Networks for Zero-Shot Learning
- [ ] [CVPR 2018] Temporal Hallucinating for Action Recognition with Few Still Images
- [ ] [ECCV 2018] Dynamic Conditional Networks for FewShot Learning
- [ ] [Nature 子刊 MI 2018] Continuous Learning of Context-dependent Processing in Neural Networks
- [ ] [CVPR 2019] Generating Classification Weights with GNN Denoising Autoencoders for Few-Shot Learning
- [ ] [CVPR 2019] Large-Scale Few-Shot Learning- Knowledge Transfer With Class Hierarchy
- [ ] [CVPR 2019] Spot and Learn A Maximum-Entropy Patch Sampler for Few-Shot Image Classification
- [ ] [CVPR 2019] Meta-Transfer Learning for Few-Shot Learning
- [ ] [CVPR 2019] Image Deformation Meta-Networks for One-Shot Learning
- [ ] [CVPR 2019] Generalized Zero- and Few-Shot Learning via Aligned Variational Autoencoders
- [ ] [NIPS 2019] ([code](https://github.com/apple2373/MetaIRNet)) Metal-Reinforced Synthetic Data for One-Shot Fine-Grained Visual Recognition


## Image Classification
### Summary
- [x] [arXiv 2019] Generalizing from a Few Examples A Survey on Few-Shot Learning
- [x] [ICLR 2019] A Closer Look At Few-shot Classification


### Optimize Based Few-shot Learning (Trying to generate Classifiers' parameters)
- [x] [CVPR 2017] Learning to Learn Image Classifiers with Visual Analogy
- [x] [ICLR 2017 Meta-learner LSTM Ravi] Optimization as a Model for Few-shot Learning
    * Use LSTM to generate classifier's parameters

- [x] [ICLR 2018 SNAIL] A Simple Neural Attentive Meta- Learner
    * Improve the Meta-Learner LSTM, by adding temporal convolution and caual attention to the network.

- [x] [CVPR 2018] Dynamic Few-Shot Visual Learning without Forgetting
- [x] [CVPR 2018] Low-Shot Learning with Imprinted Weights
    * Passing, generate weights for classifier. (I DONOT like it, the exp only compare to matching networks and "Generative + classifier")

- [x] [CVPR 2019] Dense Classification and Implanting for Few-Shot Learning
- [x] [ICML 2019] LGM-Net: Learning to Generate Matching Networks for Few shot Learning
- [x] [CVPR 2019] TAFE-Net- Task-Aware Feature Embeddings for Low Shot Learning
    * Use a meta-learner to generate parameters for the feature extractor

- [x] [NIPS 2019] Incremental Few-Shot Learning with Attention Attractor Networks
    * Using normal way to pretrain the backbone on the base classes, then using the base class weights to fintune the classifier on the few-shot episodic network.
    * Achieve the normal

- [x] [ICLR 2019 LEO Vinyals] (RECOMMENDED!) Meta-learning with latent embedding optimization
    * High dimensional problem is hard to solve in the low-data circumstances, so this work try to bypass the limitations by learning a data-dependent latent low-dimensional latent space of model parameters.


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
    * Using Hierarchical Bayesian Model after extracted features. Which is similar to build the category graph method in IJCAI 2019.

- [x] [ICML 2015] Siamese Neural Networks for One-Shot Image Recognition
- [x] [NIPS 2016] Matching Networks for One Shot Learning
    * This methods cast the problem of one-shot learning within the set-to-set framework

- [x] [NIPS 2017]  (RECOMMENDED!) Prototypical Networks for Few-shot Learning
- [x] [CVPR 2018] Learning to Compare：Relation Network for Few-Shot Learning
    * Change metric functions to CNNs
    * Provide a clean framework that elegantly encompasses both few and zero-shot learning.


##### Semi-Supervised
- [x] [ICLR 2018 Ravi] Meta-Learning for Semi-Supervised Few-Shot Classification
    * Using soft K-means to refine the prototypes, then using varient ways(training methods) to eliminate the outline points.
    * Create new datasets - tiredImagenet

- [x] [CVPR 2018] (RECOMMENDED!) Low-Shot Learning With Large-Scale Diffusion
- [x] [ICLR 2019] Learning to Propagate Labels-transductive Propagation Network for Few-shot Learning
- [x] [CVPR 2019] Edge-Labeling Graph Neural Network for Few-shot Learning


##### Supervised
- [x] [NIPS 2018] (RECOMMENDED!) TADAM-Task dependent adaptive metric for improved few-shot learning 
- [x] [CVPR 2019] Finding Task-Relevant Features for Few-Shot Learning by Category Traversal
- [x] [ICML 2019] (RECOMMENDED!) TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning
- [x] [ICML 2019] (RECOMMENDED!) Infinite Mixture Prototypes for Few-shot Learning
    * Point out that data distribution for one class are not uni-model (Verify in my experiments too).
    * (Clustering methods) Semi-Supervised methods for prototypical networks. Show this methods even suit for unsupervised situations(protentially).
    * Improve on Alphabets dataset, remain or improve on omniglot and mini-imagenet.

- [x] [ICCV 2019] Few-Shot Learning with Global Class Representations
    * Synthesis new samples to elleviate the data imbalance problem between Base and Novel Classes.
    * During training, compute two losses, one is the original losses, the other is the score for the whole classes including noval classes.

- [x] [ICCV 2019] (RECOMMENDED!) TASK2VEC- Task Embedding for Meta-Learning
    * Use Fisher information matrix to judge which backbone is suitable for current task.

- [x] [IJCAI 2019] Prototype Propagation Networks (PPN) for Weakly-supervised Few-shot Learning on Category Graph
    * Maually build an category graph, then add parents label's class represention into the child class representations.

- [x] [CVPR 2019] (RECOMMENDED) Variational Prototyping-Encoder- One-Shot Learning with Prototypical Images
    * Use encoder to translate the real images to abstract prototypes, such as painted traffic signs, then compare query and sample in the prototypes latent space.

- [x] [NIPS 2019] Adaptive Cross-Modal Few-shot Learning
    * Using texture information to enhance the performance, which reach a comparable result on mini-imagenet
    * Perform well on 1-shot rather than 5-shot or 10-shot

- [x] [CVPR 2019] Baby steps towards few-shot learning with multiple semantics
    * Show 4.5 years old baby perform 70% on 1-shot case, adult achieve 99%.
    * Add multi-semantic into the task.
    * However on 5-shot case LEO perform exceed both this paper and the paper above with no semantics information.
    * For 1-shot case, this method achieve 67.2% +- 0.4% compare to 70% of human baby performance.

- [x] [CVPR 2019] Few-Shot Learning with Localization in Realistic Settings
    * Locate the object in the images first, then classify them.
    * Classify in real-world images, somehow not interesting.

- [x] [NIPS 2019] Cross Attention Network for Few-shot Classification
    * Learn a attention(mask) to pay more attention on the part of the images
    * Add transductive inference part
    * Pretty good result on mini-imagenet 80.64 +- 0.35% under ResNet-12 (16 conv layers)

- [x] [CVPR 2019] Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning
    * Calculating the similarity between query and class represent feature in feature level, rather than instance level. It seperate original feature in $m$ part and then compute the similarity to the K-nearst class partial features.
    * Good Result on mini-ImageNet 71.02 ± 0.64% with Conv4_64F.


### Special (such as Architecture?)
#### External Memory
- [x] [ICML 2016] Meta-Learning with Memory-Augmented Neural Networks

    This work lead NTM into the image classification, technically, this work should not belong to the few-shot problems.
    This method can identify the image labels, even the true label of current image are inputed along with the next image.

- [x] [CVPR 2018] Memory Matching Networks for One-Shot Image Recognition
- [x] [ICLR 2019] Adaptive Posterior Learning-Few-Shot Learning with a Surprise-Based Memory Module

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

    Label the query set for the first run, then retrain the model with the pesudo label for the second run. (Simple but effective)


### Results in Datasets
#### [Omniglot](https://github.com/brendenlake/omniglot)
#### [mini-Imagenet](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning)

| Years | Methods              | Backbone | 5-way 1-shot    | 5-way 5-shot    |
|-------|----------------------|----------|-----------------|-----------------|
| 2016  | Matching Network     | Conv4    | 43.56 +- 0.84%  | 55.31% +- 0.73% |
| 2017  | MAML                 | Conv4    | 48.7% +- 1.84%  | 63.15% +- 0.91% |
| 2017  | Prototypical Network | Conv4    | 49.42% +- 0.78% | 68.20% +- 0.66% |
| 2018  | Relation Network     | Conv4    | 50.44% +- 0.82% | 65.32% +- 0.70% |

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
- [x] [AAAI 2019] Attention-based Multi-Context Guiding for Few-Shot Semantic Segmentation

    Utilize the output of the different layers between query branch and support branch to gain more context informations.

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

- [x] [arXiv 2019] Few-Shot Text Classification with Induction Network

    Introduce dynamic routing to generate better class representations. One real industrial project.
