# 目录
<!-- vim-markdown-toc GitLab -->

<<<<<<< HEAD
- [目录](#%e7%9b%ae%e5%bd%95)
- [Introduction](#introduction)
- [ToDo](#todo)
  - [ReadLater](#readlater)
    - [Generative](#generative)
    - [Optimize](#optimize)
  - [Focus Reading](#focus-reading)
- [ReadLater](#readlater-1)
- [Sorting](#sorting)
  - [Sorting](#sorting-1)
- [Image Classification](#image-classification)
  - [Summary](#summary)
  - [Optimize Based Few-shot Learning (Trying to generate Classifiers' parameters)](#optimize-based-few-shot-learning-trying-to-generate-classifiers-parameters)
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
=======
* [Introduction](#introduction)
  * [ToDo](#todo)
  * [No Paper](#no-paper)
* [Image Classification](#image-classification)
  * [Summary](#summary)
  * [Optimize Based Few-shot Learning](#optimize-based-few-shot-learning)
  * [Generative Based Few-shot Learning](#generative-based-few-shot-learning)
    * [Summary](#summary-1)
    * [Papers](#papers)
  * [Metric Based Few-shot Learning](#metric-based-few-shot-learning)
    * [Traditional](#traditional)
    * [Semi-Supervised](#semi-supervised)
    * [Supervised](#supervised)
  * [Special](#special)
    * [Unsorted](#unsorted)
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
>>>>>>> upstream/master

<!-- vim-markdown-toc -->
# Introduction

This Repo is used to Collect Few-shot Learning Area Papers, welcome to supplement in the Issues.

## ToDo
- [ ] (WIP)Add one comment to each works
- [ ] (WIP)Add results tables for all works
- [ ] Add paper link and opensource code for each works

<<<<<<< HEAD
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

# ReadLater
- [ ] [NIPS 2018] Stacked Semantics-Guided Attention Model for Fine-Grained Zero-Shot Learning
- [ ] [NIPS 2018] Generalized Zero-Shot Learning with Deep Calibration Network
- [ ] [CVPR 2017] AGA - Attribute-Guided Augmentation



# Sorting
- [ ] [arXiv 2018 REPTILE] ([paper](https://arxiv.org/pdf/1803.02999.pdf)) On First-Order Meta-Learning Algorithms

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
- [ ] [arXiv 2019] Centroid Networks for Few-Shot Clustering and Unsupervised Few-Shot Classification
- [ ] [CVPR 2018] ([paper](https://arxiv.org/pdf/1902.08605)) Temporal Hallucinating for Action Recognition with Few Still Images
- [ ] [Nature 子刊 MI 2018] ([paper](https://arxiv.org/pdf/1810.01256.pdf)) Continuous Learning of Context-dependent Processing in Neural Networks
- [ ] [CVPR 2019] ([paper](https://arxiv.org/pdf/1905.01102.pdf) [code](https://github.com/gidariss/wDAE_GNN_FewShot)) Generating Classification Weights with GNN Denoising Autoencoders for Few-Shot Learning
- [ ] [CVPR 2019] ([paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chu_Spot_and_Learn_A_Maximum-Entropy_Patch_Sampler_for_Few-Shot_Image_CVPR_2019_paper.pdf)) Spot and Learn A Maximum-Entropy Patch Sampler for Few-Shot Image Classification
- [ ] [CVPR 2019] ([paper](https://arxiv.org/pdf/1812.02391v2.pdf)) Meta-Transfer Learning for Few-Shot Learning
- [ ] [CVPR 2019 IDeMe-Net] ([paper](https://arxiv.org/pdf/1905.11641.pdf) [code](https://github.com/tankche1/IDeMe-Net.)) Image Deformation Meta-Networks for One-Shot Learning
- [ ] [CVPR 2019] ([paper](https://arxiv.org/pdf/1812.01784.pdf) [code](https://github.com/edgarschnfld/CADA-VAE-PyTorc)) Generalized Zero- and Few-Shot Learning via Aligned Variational Autoencoders
=======
## No Paper
>>>>>>> upstream/master
- [ ] [NIPS 2019] ([code](https://github.com/apple2373/MetaIRNet)) Metal-Reinforced Synthetic Data for One-Shot Fine-Grained Visual Recognition

# Image Classification
## Summary
- [x] [arXiv 2019] ([paper](https://arxiv.org/pdf/1904.05046.pdf)) Generalizing from a Few Examples A Survey on Few-Shot Learning
- [x] [ICLR 2019] ([paper](https://arxiv.org/pdf/1904.04232) [code](https://github.com/wyharveychen/CloserLookFewShot)) A Closer Look At Few-shot Classification


<<<<<<< HEAD
## Optimize Based Few-shot Learning (Trying to generate Classifiers' parameters)
=======
## Optimize Based Few-shot Learning
>>>>>>> upstream/master
- [x] [CVPR 2017] ([paper](https://arxiv.org/pdf/1710.06177.pdf)) Learning to Learn Image Classifiers with Visual Analogy
- [x] [ICLR 2017 Meta-learner LSTM Ravi] ([paper](https://openreview.net/pdf?id=rJY0-Kcll&source=post_page---------------------------) [code](https://github.com/twitter/meta-learning-lstm.)) Optimization as a Model for Few-shot Learning
    * Use LSTM to generate classifier's parameters

<<<<<<< HEAD
=======
- [x] [arXiv 2018 REPTILE] ([paper](https://arxiv.org/pdf/1803.02999.pdf)) On First-Order Meta-Learning Algorithms
>>>>>>> upstream/master
- [x] [ICLR 2018 SNAIL] ([paper](https://arxiv.org/pdf/1707.03141.pdf)) A Simple Neural Attentive Meta- Learner
    * Improve the Meta-Learner LSTM, by adding temporal convolution and caual attention to the network.

- [x] [CVPR 2018] ([paper](https://arxiv.org/pdf/1804.09458.pdf) [code](https://github.com/gidariss/FewShotWithoutForgetting)) Dynamic Few-Shot Visual Learning without Forgetting
- [x] [CVPR 2018] ([paper](https://arxiv.org/pdf/1712.07136.pdf)) Low-Shot Learning with Imprinted Weights
    * Passing, generate weights for classifier. (I DONOT like it, the exp only compare to matching networks and "Generative + classifier")

- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1903.05050.pdf)) Dense Classification and Implanting for Few-Shot Learning
- [x] [ICML 2019] ([paper](https://arxiv.org/pdf/1905.06331.pdf) [code](https://github.com/likesiwell/LGM-Net/)) LGM-Net: Learning to Generate Matching Networks for Few shot Learning
- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1904.05967.pdf) [code](https://github.com/ucbdrive/tafe-net)) TAFE-Net- Task-Aware Feature Embeddings for Low Shot Learning
    * Use a meta-learner to generate parameters for the feature extractor

- [x] [NIPS 2019] ([paper](https://arxiv.org/pdf/1810.07218.pdf) [code](https://github.com/renmengye/inc-few-shot-attractor-public)) Incremental Few-Shot Learning with Attention Attractor Networks
    * Using normal way to pretrain the backbone on the base classes, then using the base class weights to fintune the classifier on the few-shot episodic network.
    * Achieve the normal

- [x] [ICLR 2019 LEO Vinyals] (RECOMMENDED!) ([paper](https://arxiv.org/pdf/1807.05960.pdf) [code](https://github.com/deepmind/leo.)) Meta-learning with latent embedding optimization
    * High dimensional problem is hard to solve in the low-data circumstances, so this work try to bypass the limitations by learning a data-dependent latent low-dimensional latent space of model parameters.

- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1905.01102.pdf) [code](https://github.com/gidariss/wDAE_GNN_FewShot)) Generating Classification Weights with GNN Denoising Autoencoders for Few-Shot Learning
    * Little better than LEO


## Generative Based Few-shot Learning
### Summary
- [x] [NIPS 2018 Bengio] ([paper](https://papers.nips.cc/paper/7504-metagan-an-adversarial-approach-to-few-shot-learning.pdf)) MetaGAN An Adversarial Approach to Few-Shot Learning


### Papers
- [x] [ICCV 2017] ([paper](https://arxiv.org/pdf/1606.02819.pdf) [code](https://github.com/facebookresearch/low-shot-shrink-hallucinate)) Low-shot Visual Recognition by Shrinking and Hallucinating Features
- [x] [CVPR 2018] ([paper](https://arxiv.org/pdf/1801.05401.pdf)) Low-Shot Learning from Imaginary Data
- [x] [NIPS 2018] ([paper](https://arxiv.org/pdf/1810.11730.pdf)) Low-shot Learning via Covariance-Preserving Adversarial Augmentation Networks
<<<<<<< HEAD
=======
- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1812.01784.pdf) [code](https://github.com/edgarschnfld/CADA-VAE-PyTorc)) Generalized Zero- and Few-Shot Learning via Aligned Variational Autoencoders
    * Aiming at cross modal situations
    * Using encode image x and class descriptor y, then pull the mean and variance of them together by a loss.
- [x] [CVPR 2019 IDeMe-Net] ([paper](https://arxiv.org/pdf/1905.11641.pdf) [code](https://github.com/tankche1/IDeMe-Net.)) Image Deformation Meta-Networks for One-Shot Learning
    * This paper assumes that deformed images may not be visually realistic, they still maintain critical semantic information.
    * Pretty good at one-shot on mini-imagenet(59.14%)
>>>>>>> upstream/master


## Metric Based Few-shot Learning
### Traditional
- [x] [ICML 2012] One-Shot Learning with a Hierarchical Nonparametric Bayesian Model
    * Using Hierarchical Bayesian Model after extracted features. Which is similar to build the category graph method in IJCAI 2019.

- [x] [ICML 2015] Siamese Neural Networks for One-Shot Image Recognition
- [x] [NIPS 2016] Matching Networks for One Shot Learning
    * This methods cast the problem of one-shot learning within the set-to-set framework

- [x] [NIPS 2017]  (RECOMMENDED!) Prototypical Networks for Few-shot Learning
- [x] [CVPR 2018] Learning to Compare：Relation Network for Few-Shot Learning
    * Change metric functions to CNNs
    * Provide a clean framework that elegantly encompasses both few and zero-shot learning.


### Semi-Supervised
- [x] [ICLR 2018 Ravi] ([paper](https://arxiv.org/pdf/1803.00676.pdf) [code](https://github.com/renmengye/few-shot-ssl-public)) Meta-Learning for Semi-Supervised Few-Shot Classification
    * Using soft K-means to refine the prototypes, then using varient ways(training methods) to eliminate the outline points.
    * Create new datasets - tiredImagenet

- [x] [CVPR 2018] (RECOMMENDED!) Low-Shot Learning With Large-Scale Diffusion
- [x] [ICLR 2019] ([paper](https://arxiv.org/pdf/1805.10002.pdf)) Learning to Propagate Labels-transductive Propagation Network for Few-shot Learning
- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1905.01436.pdf)) Edge-Labeling Graph Neural Network for Few-shot Learning


### Supervised
- [x] [NIPS 2018] (RECOMMENDED!) TADAM-Task dependent adaptive metric for improved few-shot learning 
<<<<<<< HEAD
- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1905.11116.pdf) [code](https://github.com/Clarifai/few-shot-ctm.)) Finding Task-Relevant Features for Few-Shot Learning by Category Traversal
- [x] [ICML 2019] (RECOMMENDED!) ([paper](https://arxiv.org/pdf/1905.06549.pdf)) TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning
- [x] [ICML 2019] (RECOMMENDED!) Infinite Mixture Prototypes for Few-shot Learning
=======
    * In every task, use task representations to finetune the output of each Conv Blocks, like BN functionally.

- [x] [ECCV 2018] ([paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Fang_Zhao_Dynamic_Conditional_Networks_ECCV_2018_paper.pdf)) Dynamic Conditional Networks for FewShot Learning
    * Basically same as TADAM(using task representation to finetune the backbones), it use a conditional feature to influence the output of conv layers (Linear combination).

- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1905.11116.pdf) [code](https://github.com/Clarifai/few-shot-ctm.)) Finding Task-Relevant Features for Few-Shot Learning by Category Traversal
- [x] [ICML 2019] (RECOMMENDED!) ([paper](https://arxiv.org/pdf/1905.06549.pdf)) TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning
- [x] [ICML 2019] ([paper](https://arxiv.org/pdf/1902.04552.pdf)) (RECOMMENDED!) Infinite Mixture Prototypes for Few-shot Learning
>>>>>>> upstream/master
    * Point out that data distribution for one class are not uni-model (Verify in my experiments too).
    * (Clustering methods) Semi-Supervised methods for prototypical networks. Show this methods even suit for unsupervised situations(protentially).
    * Improve on Alphabets dataset, remain or improve on omniglot and mini-imagenet.

- [x] [ICCV 2019] ([paper](https://arxiv.org/pdf/1908.05257)) Few-Shot Learning with Global Class Representations
    * Synthesis new samples to elleviate the data imbalance problem between Base and Novel Classes.
    * During training, compute two losses, one is the original losses, the other is the score for the whole classes including noval classes.


- [x] [IJCAI 2019] ([paper](https://arxiv.org/pdf/1905.04042) [code](https://github.com/liulu112601/PPN)) Prototype Propagation Networks (PPN) for Weakly-supervised Few-shot Learning on Category Graph
    * Maually build an category graph, then add parents label's class represention into the child class representations.

- [x] [CVPR 2019] (RECOMMENDED) ([paper](https://arxiv.org/pdf/1904.08482.pdf) [code](https://github.com/mibastro/VPE)) Variational Prototyping-Encoder- One-Shot Learning with Prototypical Images
    * Use encoder to translate the real images to abstract prototypes, such as painted traffic signs, then compare query and sample in the prototypes latent space.

- [x] [NIPS 2019] ([paper](https://arxiv.org/pdf/1902.07104.pdf)) Adaptive Cross-Modal Few-shot Learning
    * Using texture information to enhance the performance, which reach a comparable result on mini-imagenet
    * Perform well on 1-shot rather than 5-shot or 10-shot

<<<<<<< HEAD
=======
- [x] [CVPR 2019] ([paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chu_Spot_and_Learn_A_Maximum-Entropy_Patch_Sampler_for_Few-Shot_Image_CVPR_2019_paper.pdf)) Spot and Learn A Maximum-Entropy Patch Sampler for Few-Shot Image Classification
    * Sample parts of the image to from the batch to represent the class. 
    * One-shot not pretty good(51%)

>>>>>>> upstream/master
- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1906.01905.pdf)) Baby steps towards few-shot learning with multiple semantics
    * Show 4.5 years old baby perform 70% on 1-shot case, adult achieve 99%.
    * Add multi-semantic into the task.
    * However on 5-shot case LEO perform exceed both this paper and the paper above with no semantics information.
    * For 1-shot case, this method achieve 67.2% +- 0.4% compare to 70% of human baby performance.

- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1904.08502)) Few-Shot Learning with Localization in Realistic Settings
    * Locate the object in the images first, then classify them.
    * Classify in real-world images, somehow not interesting.

- [x] [NIPS 2019] ([paper](https://arxiv.org/pdf/1910.07677.pdf)) Cross Attention Network for Few-shot Classification
    * Learn a attention(mask) to pay more attention on the part of the images
    * Add transductive inference part
    * Pretty good result on mini-imagenet 80.64 +- 0.35% under ResNet-12 (16 conv layers)

- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1903.12290.pdf)) Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning
    * Calculating the similarity between query and class represent feature in feature level, rather than instance level. It seperate original feature in m part and then compute the similarity to the K-nearst class partial features.
    * Good Result on mini-ImageNet 71.02 ± 0.64% with Conv4_64F.

- [x] [CVPR 2019] Large-Scale Few-Shot Learning- Knowledge Transfer With Class Hierarchy
    * Aiming at learning large-scale problem, not just on 5 novel class.
    * Using the Class Names embeddings(text embedding) to form a class hierarchy.
    * Get a pretter higher result than existing methods.

- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1812.02391v2.pdf)) Meta-Transfer Learning for Few-Shot Learning
    * Not like it, for the results are not significant, nearly no improve on 5 way 5 shot on mini-ImageNet.

- [x] [CVPR 2018] ([paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Temporal_Hallucinating_for_CVPR_2018_paper.pdf)) Temporal Hallucinating for Action Recognition with Few Still Images
    * Attempt to recall cues from relevant action videos.
    * Maybe good at one-shot, not worse than the baseline in 5-shot and 10-shot scenarios.

## Special
### Unsorted
- [x] [Nature 子刊 MI 2018] ([paper](https://arxiv.org/pdf/1810.01256.pdf)) Continuous Learning of Context-dependent Processing in Neural Networks
    * During training a network consecutively for different tasks, OWNs weights are only allowed to be modified in the direction orthogonal to the subspace spanned by all inputs on which the network has been trained (termed input space hereafter). This ensures that new learning processes will not interfere with the learned tasks

### External Memory
- [x] [ICML 2016] Meta-Learning with Memory-Augmented Neural Networks

    This work lead NTM into the image classification, technically, this work should not belong to the few-shot problems.
    This method can identify the image labels, even the true label of current image are inputed along with the next image.

- [x] [CVPR 2018] ([paper](https://arxiv.org/pdf/1804.08281.pdf)) Memory Matching Networks for One-Shot Image Recognition
- [x] [ICLR 2019] ([paper](https://arxiv.org/pdf/1902.02527.pdf) [code](https://github.com/cogentlabs/apl.)) Adaptive Posterior Learning-Few-Shot Learning with a Surprise-Based Memory Module

### Architecture
- [x] [ICML 2017] Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1805.07722.pdf)) Task-Agnostic Meta-Learning for Few-shot Learning

    A training method force model to learn a unbiased initial model without over-performing on some particular tasks.

### Task Representation and Measure
- [x] [ICCV 2019] ([paper](https://arxiv.org/pdf/1902.03545.pdf)) (RECOMMENDED!) TASK2VEC- Task Embedding for Meta-Learning
    * Use Fisher information matrix to judge which backbone is suitable for current task.

### Multi Label Image Classification
- [x] [CVPR 2019 oral] ([paper](https://arxiv.org/pdf/1902.09811.pdf)) LaSO-Label-Set Operations networks for multi-label few-shot learning

### Add Additional Informations
- [x] [ICCV 2019] ([paper](https://arxiv.org/pdf/1812.09213.pdf) [code](https://sites.google.com/view/comprepr/home)) Learning Compositional Representations for Few-Shot Recognition

    Add additional annotations to the classes.

- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1904.03472.pdf)) Few-shot Learning via Saliency-guided Hallucination of Samples

    Form segmentations and mix up, aiming at eliminates the back ground noise.

- [x] [ICCV 2019] ([paper](https://arxiv.org/pdf/1906.05186.pdf)) Boosting Few-Shot Visual Learning with Self-Supervision

    Self-supervision means to rotate itself, and compute two losses.

### Self-training

- [x] [NIPS 2019] ([paper](https://arxiv.org/pdf/1906.00562.pdf)) Learning to Self-Train for Semi-Supervised Few-Shot Classification

    Label the query set for the first run, then retrain the model with the pesudo label for the second run. (Simple but effective)


## Results in Datasets
### [Omniglot](https://github.com/brendenlake/omniglot)
### [mini-Imagenet](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning)

| Years | Methods              | Backbone | 5-way 1-shot    | 5-way 5-shot    |
|-------|----------------------|----------|-----------------|-----------------|
| 2016  | Matching Network     | Conv4    | 43.56 +- 0.84%  | 55.31% +- 0.73% |
| 2017  | MAML                 | Conv4    | 48.7% +- 1.84%  | 63.15% +- 0.91% |
| 2017  | Prototypical Network | Conv4    | 49.42% +- 0.78% | 68.20% +- 0.66% |
| 2018  | Relation Network     | Conv4    | 50.44% +- 0.82% | 65.32% +- 0.70% |

### [tiredImagenet](https://arxiv.org/abs/1803.00676)
### [Imagenet](http://image-net.org)
### [CUB 2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

# Object Detection
<<<<<<< HEAD
=======
- [x] [CVPR 2017] ([paper](https://arxiv.org/pdf/1612.02559.pdf) [code](https://github.com/rkwitt/GuidedAugmentation)) AGA - Attribute-Guided Augmentation
    * Using external depth and pose informations
>>>>>>> upstream/master
- [x] [CVPR 2019] ([paper](https://arxiv.org/abs/1806.04728)) RepMet-Representative-based Metric Learning for Classification and Few-shot Object Detection
- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1903.09372.pdf)) Few-shot Adaptive Faster R-CNN
- [x] [CVPR 2019] Feature Selective Anchor-Free Module for Single-Shot Object Detection
- [x] [ICCV 2019] Few-shot Object Detection via Feature Reweighting

# Segementation
- [x] [CVPR 2019] CANet- Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning
- [x] [AAAI 2019] ([paper](https://www.researchgate.net/publication/335296764_Attention-Based_Multi-Context_Guiding_for_Few-Shot_Semantic_Segmentation)) Attention-based Multi-Context Guiding for Few-Shot Semantic Segmentation
    * Utilize the output of the different layers between query branch and support branch to gain more context informations.

- [x] [arXiv 2019] AMP-Adaptive Masked Proxies for Few-Shot Segmentation
    * Not sure result in this area.

- [x] [CVPR 2019] ([paper](https://arxiv.org/pdf/1902.11123.pdf) [code](https://github.com/MSiam/AdaptiveMaskedProxies.)) SAR Image Classification Using Few-shot Cross-domain Transfer Learning

# Generative Model
- [x] [ICCV 2019] Few-Shot Unsupervised Image-to-Image Translation
- [x] [CVPR 2018] Multi-Content GAN for Few-Shot Font Style Transfer

# Domain Adaptation
- [x] [NIPS 2017] Few-Shot Adversarial Domain Adaptation
- [x] [ICCV 2019] Bidirectional One-Shot Unsupervised Domain Mapping

# Reinforcement Learning
- [x] [ICML 2019] Few-Shot Intent Inference via Meta-Inverse Reinforcement Learning

# Visual Tracking
- [x] [ICCV 2019] Deep Meta Learning for Real-Time Target-Aware Visual Tracking

# Others
- [x] [IJCAI 2019] Incremental Few-Shot Learning for Pedestrian Attribute Recognition
- [x] [AAAI 2018] AffinityNet- Semi-supervised Few-shot Learning for Disease Type Prediction
    * Use few-shot method to enhance oringal disease type prediction

- [x] [arXiv 2019] ([paper](https://arxiv.org/pdf/1902.10482.pdf)) Few-Shot Text Classification with Induction Network
    * Introduce dynamic routing to generate better class representations. One real industrial project.
