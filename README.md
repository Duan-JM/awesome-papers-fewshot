Awesome Papers Few-shot focus on collecting paper published on top conferences in Few-shot learning area,
hoping that this cut some time costing for beginners. Morever we also glad to see this repo can be a virtual online seminar,
which can be a home to all researchers who have the enthusiasm to exchange interesting ideas.

Awesome Papers Few-shot 收集了近几年定会中与小样本学习相关的论文，并简单的进行了类别划分与整理。
一方面，我们希望这个仓库能够帮助广大希望入坑小样本学习的同胞减少入坑搜集论文的时间成本。另一方面，
我们也希望这里能称为研究小样本学习的同胞们互相交流有趣想法的一个小平台。

The papers collected in this repo are manually selected by myself, I am hoping that more researchers interested in this area can maintain this repo together.

仓库中收藏的论文均为我本人从历年顶会中手动挑选并阅读过和小样本学习相关的论文，也希望能有广大的同行来共同维护它。
（注意：部分深入解释 Meta-Learning 的论文并未收入到此仓库中，有兴趣的朋友可以发 issue 一起讨论）。

# 目录
<!-- vim-markdown-toc GitLab -->

* [Image Classification](#image-classification)
  * [Parameter Optimize Based Few-shot Learning](#parameter-optimize-based-few-shot-learning)
      * [Papers](#papers)
  * [Generative Based Few-shot Learning](#generative-based-few-shot-learning)
      * [Papers](#papers-1)
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
      * [mini-Imagenet](#mini-imagenet)
* [More Direction](#more-direction)
    * [Object Detection](#object-detection)
    * [Segementation](#segementation)
    * [Generative Model](#generative-model)
    * [Domain Adaptation](#domain-adaptation)
    * [Reinforcement Learning](#reinforcement-learning)
    * [Visual Tracking](#visual-tracking)
    * [Others](#others)
* [Other Awesome Resources](#other-awesome-resources)
    * [Relevant Awesome Datasets Repo](#relevant-awesome-datasets-repo)
    * [Relevant Awesome Few-shot PlayGround Repo](#relevant-awesome-few-shot-playground-repo)
    * [Relevant Awesome Blogs](#relevant-awesome-blogs)
* [How to recommend a paper](#how-to-recommend-a-paper)
* [Main Contributors](#main-contributors)

<!-- vim-markdown-toc -->

# Image Classification
- [arXiv 2019] ([paper](https://arxiv.org/pdf/1904.05046.pdf)) Generalizing from a Few Examples A Survey on Few-Shot Learning
- [ICLR 2019] ([paper](https://arxiv.org/pdf/1904.04232) [code](https://github.com/wyharveychen/CloserLookFewShot)) A Closer Look At Few-shot Classification
- [arXiv 2019] ([paper](https://arxiv.org/pdf/1909.02729.pdf)) A Baseline for Few-shot Image Classification


## Parameter Optimize Based Few-shot Learning
**One line descriptions:** Generate parameters for the classifier or finetune part of the models

#### Papers
- [ICLR 2017 Meta-learner LSTM Ravi] ([paper](https://openreview.net/pdf?id=rJY0-Kcll&source=post_page---------------------------) [code](https://github.com/twitter/meta-learning-lstm.)) Optimization as a Model for Few-shot Learning
    * Use LSTM to generate classifier's parameters

- [arXiv 2018 REPTILE] ([paper](https://arxiv.org/pdf/1803.02999.pdf)) On First-Order Meta-Learning Algorithms
- [ICLR 2018 SNAIL] ([paper](https://arxiv.org/pdf/1707.03141.pdf)) A Simple Neural Attentive Meta- Learner
    * Improve the Meta-Learner LSTM, by adding temporal convolution and caual attention to the network.

- [CVPR 2018] ([paper](https://arxiv.org/pdf/1804.09458.pdf) [code](https://github.com/gidariss/FewShotWithoutForgetting)) Dynamic Few-Shot Visual Learning without Forgetting
- [CVPR 2018] ([paper](https://arxiv.org/pdf/1712.07136.pdf)) Low-Shot Learning with Imprinted Weights
    * Passing, generate weights for classifier. (I DONOT like it, the exp only compare to matching networks and "Generative + classifier")

- [CVPR 2019] (RECOMMENDED!) ([paper](https://arxiv.org/pdf/1710.06177.pdf)) Learning to Learn Image Classifiers with Visual Analogy
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1903.05050.pdf)) Dense Classification and Implanting for Few-Shot Learning
- [ICML 2019] ([paper](https://arxiv.org/pdf/1905.06331.pdf) [code](https://github.com/likesiwell/LGM-Net/)) LGM-Net: Learning to Generate Matching Networks for Few shot Learning
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1904.05967.pdf) [code](https://github.com/ucbdrive/tafe-net)) TAFE-Net- Task-Aware Feature Embeddings for Low Shot Learning
    * Use a meta-learner to generate parameters for the feature extractor

- [NIPS 2019] ([paper](https://arxiv.org/pdf/1810.07218.pdf) [code](https://github.com/renmengye/inc-few-shot-attractor-public)) Incremental Few-Shot Learning with Attention Attractor Networks
    * Using normal way to pretrain the backbone on the base classes, then using the base class weights to fintune the classifier on the few-shot episodic network.
    * Achieve the normal

- [ICLR 2019 LEO Vinyals] (RECOMMENDED!) ([paper](https://arxiv.org/pdf/1807.05960.pdf) [code](https://github.com/deepmind/leo)) Meta-learning with latent embedding optimization
    * High dimensional problem is hard to solve in the low-data circumstances, so this work try to bypass the limitations by learning a data-dependent latent low-dimensional latent space of model parameters.

- [CVPR 2019] ([paper](https://arxiv.org/pdf/1905.01102.pdf) [code](https://github.com/gidariss/wDAE_GNN_FewShot)) Generating Classification Weights with GNN Denoising Autoencoders for Few-Shot Learning
    * Little better than LEO

- [NIPS 2018] Delta-encoder: an effective sample synthesis method for few-shot object recognition
- [ICLR 2019] Meta-learning with differentiable closed-form solvers
    * Teach to use tradional machining learning methods
    * Most likely no good than LEO

- [ICML 2019] Fast Context Adaptation via Meta-Learning
    * Update partial parameters


## Generative Based Few-shot Learning
**One line descriptions:** Generate features to expasion small datasets to large datasets, then fintune.

- [NIPS 2018 Bengio] ([paper](https://papers.nips.cc/paper/7504-metagan-an-adversarial-approach-to-few-shot-learning.pdf)) MetaGAN An Adversarial Approach to Few-Shot Learning


#### Papers
- [ICCV 2017] ([paper](https://arxiv.org/pdf/1606.02819.pdf) [code](https://github.com/facebookresearch/low-shot-shrink-hallucinate)) Low-shot Visual Recognition by Shrinking and Hallucinating Features
- [CVPR 2018] ([paper](https://arxiv.org/pdf/1801.05401.pdf)) Low-Shot Learning from Imaginary Data
- [NIPS 2018] ([paper](https://arxiv.org/pdf/1810.11730.pdf)) Low-shot Learning via Covariance-Preserving Adversarial Augmentation Networks
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1812.01784.pdf) [code](https://github.com/edgarschnfld/CADA-VAE-PyTorc)) Generalized Zero- and Few-Shot Learning via Aligned Variational Autoencoders
    * Aiming at cross modal situations
    * Using encode image x and class descriptor y, then pull the mean and variance of them together by a loss.
- [CVPR 2019 IDeMe-Net] ([paper](https://arxiv.org/pdf/1905.11641.pdf) [code](https://github.com/tankche1/IDeMe-Net.)) Image Deformation Meta-Networks for One-Shot Learning
    * This paper assumes that deformed images may not be visually realistic, they still maintain critical semantic information.
    * Pretty good at one-shot on mini-imagenet(59.14%)


## Metric Based Few-shot Learning
**One line descriptions:** Compute the class representation, then use metric functions to measure the similarity between query sample and each class representaions.

#### Traditional
- [ICML 2012] One-Shot Learning with a Hierarchical Nonparametric Bayesian Model
    * Using Hierarchical Bayesian Model after extracted features. Which is similar to build the category graph method in IJCAI 2019.

- [ICML 2015] Siamese Neural Networks for One-Shot Image Recognition
- [NIPS 2016] Matching Networks for One Shot Learning
    * This methods cast the problem of one-shot learning within the set-to-set framework

- [NIPS 2017]  (RECOMMENDED!) Prototypical Networks for Few-shot Learning
- [CVPR 2018] Learning to Compare：Relation Network for Few-Shot Learning
    * Change metric functions to CNNs
    * Provide a clean framework that elegantly encompasses both few and zero-shot learning.


#### Semi-Supervised
- [ICLR 2018 Ravi] ([paper](https://arxiv.org/pdf/1803.00676.pdf) [code](https://github.com/renmengye/few-shot-ssl-public)) Meta-Learning for Semi-Supervised Few-Shot Classification
    * Using soft K-means to refine the prototypes, then using varient ways(training methods) to eliminate the outline points.
    * Create new datasets - tiredImagenet

- [ICLR 2018] Few-Shot Learning with Graph Neural Networks
- [CVPR 2018] (RECOMMENDED!) Low-Shot Learning With Large-Scale Diffusion
- [ICLR 2019] ([paper](https://arxiv.org/pdf/1805.10002.pdf)) Learning to Propagate Labels-transductive Propagation Network for Few-shot Learning
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1905.01436.pdf)) Edge-Labeling Graph Neural Network for Few-shot Learning


#### Supervised
- [NIPS 2017] Few-Shot Learning Through an Information Retrieval Lens
- [NIPS 2018] (RECOMMENDED!) TADAM-Task dependent adaptive metric for improved few-shot learning 
    * In every task, use task representations to finetune the output of each Conv Blocks, like BN functionally.

- [ECCV 2018] ([paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Fang_Zhao_Dynamic_Conditional_Networks_ECCV_2018_paper.pdf)) Dynamic Conditional Networks for FewShot Learning
    * Basically same as TADAM(using task representation to finetune the backbones), it use a conditional feature to influence the output of conv layers (Linear combination).

- [CVPR 2019] ([paper](https://arxiv.org/pdf/1905.11116.pdf) [code](https://github.com/Clarifai/few-shot-ctm.)) Finding Task-Relevant Features for Few-Shot Learning by Category Traversal
- [ICML 2019] (RECOMMENDED!) ([paper](https://arxiv.org/pdf/1905.06549.pdf)) TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning
- [ICML 2019] ([paper](https://arxiv.org/pdf/1902.04552.pdf)) (RECOMMENDED!) Infinite Mixture Prototypes for Few-shot Learning
    * Point out that data distribution for one class are not uni-model (Verify in my experiments too).
    * (Clustering methods) Semi-Supervised methods for prototypical networks. Show this methods even suit for unsupervised situations(protentially).
    * Improve on Alphabets dataset, remain or improve on omniglot and mini-imagenet.

- [ICCV 2019] ([paper](https://arxiv.org/pdf/1908.05257)) Few-Shot Learning with Global Class Representations
    * Synthesis new samples to elleviate the data imbalance problem between Base and Novel Classes.
    * During training, compute two losses, one is the original losses, the other is the score for the whole classes including noval classes.


- [IJCAI 2019] ([paper](https://arxiv.org/pdf/1905.04042) [code](https://github.com/liulu112601/PPN)) Prototype Propagation Networks (PPN) for Weakly-supervised Few-shot Learning on Category Graph
    * Maually build an category graph, then add parents label's class represention into the child class representations.

- [CVPR 2019] (RECOMMENDED) ([paper](https://arxiv.org/pdf/1904.08482.pdf) [code](https://github.com/mibastro/VPE)) Variational Prototyping-Encoder- One-Shot Learning with Prototypical Images
    * Use encoder to translate the real images to abstract prototypes, such as painted traffic signs, then compare query and sample in the prototypes latent space.

- [NIPS 2019] ([paper](https://arxiv.org/pdf/1902.07104.pdf)) Adaptive Cross-Modal Few-shot Learning
    * Using texture information to enhance the performance, which reach a comparable result on mini-imagenet
    * Perform well on 1-shot rather than 5-shot or 10-shot

- [CVPR 2019] ([paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chu_Spot_and_Learn_A_Maximum-Entropy_Patch_Sampler_for_Few-Shot_Image_CVPR_2019_paper.pdf)) Spot and Learn A Maximum-Entropy Patch Sampler for Few-Shot Image Classification
    * Sample parts of the image to form the batch to represent the class. 
    * One-shot not pretty good(51%)

- [CVPR 2019] ([paper](https://arxiv.org/pdf/1906.01905.pdf)) Baby steps towards few-shot learning with multiple semantics
    * Show 4.5 years old baby perform 70% on 1-shot case, adult achieve 99%.
    * Add multi-semantic into the task.
    * However on 5-shot case LEO perform exceed both this paper and the paper above with no semantics information.
    * For 1-shot case, this method achieve 67.2% +- 0.4% compare to 70% of human baby performance.

- [CVPR 2019] ([paper](https://arxiv.org/pdf/1904.08502)) Few-Shot Learning with Localization in Realistic Settings
    * Locate the object in the images first, then classify them.
    * Classify in real-world images, somehow not interesting.

- [NIPS 2019] ([paper](https://arxiv.org/pdf/1910.07677.pdf)) Cross Attention Network for Few-shot Classification
    * Learn a attention(mask) to pay more attention on the part of the images
    * Add transductive inference part
    * Pretty good result on mini-imagenet 80.64 +- 0.35% under ResNet-12 (16 conv layers)

- [CVPR 2019] ([paper](https://arxiv.org/pdf/1903.12290.pdf)) Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning
    * Calculating the similarity between query and class represent feature in feature level, rather than instance level. It seperate original feature in m part and then compute the similarity to the K-nearst class partial features.
    * Good Result on mini-ImageNet 71.02 ± 0.64% with Conv4_64F.

- [CVPR 2019] Large-Scale Few-Shot Learning- Knowledge Transfer With Class Hierarchy
    * Aiming at learning large-scale problem, not just on 5 novel class.
    * Using the Class Names embeddings(text embedding) to form a class hierarchy.
    * Get a pretter higher result than existing methods.

- [CVPR 2019] ([paper](https://arxiv.org/pdf/1812.02391v2.pdf)) Meta-Transfer Learning for Few-Shot Learning
    * Not like it, for the results are not significant, nearly no improve on 5 way 5 shot on mini-ImageNet.

- [CVPR 2018] ([paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Temporal_Hallucinating_for_CVPR_2018_paper.pdf)) Temporal Hallucinating for Action Recognition with Few Still Images
    * Attempt to recall cues from relevant action videos.
    * Maybe good at one-shot, not worse than the baseline in 5-shot and 10-shot scenarios.

- [NIPS 2019] Learning to Propagate for Graph Meta-Learning
    * Learns to propagate messages between prototypes of different classes on the graph, so that learning the prototype of each class benefits from the data of other related classes.
    * Attention mechanic.

- [ICCV 2019] Transductive Episodic-Wise Adaptive Metric for Few-Shot Learning

- [ICCV 2019] ([paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Hao_Collect_and_Select_Semantic_Alignment_Metric_Learning_for_Few-Shot_Learning_ICCV_2019_paper.pdf) Collect and Select: Semantic Alignment Metric Learning for Few-Shot Learning
    * Use attention to pick(Select) most relevant part to compare

- [AAAI 2019] Distribution Consistency based Covariance Metric Networks for Few Shot Learning
    * Slight improve on 1-shot compare to Relation Network, however degenerate on 5-shot compare to Protoypical Network.

- [AAAI 2019] A Dual Attention Network with Semantic Embedding for Few-shot Learning
    * Add spatial attention and task attention.

## Special
#### Unsorted
- [Nature 子刊 MI 2018] ([paper](https://arxiv.org/pdf/1810.01256.pdf)) Continuous Learning of Context-dependent Processing in Neural Networks
    * During training a network consecutively for different tasks, OWNs weights are only allowed to be modified in the direction orthogonal to the subspace spanned by all inputs on which the network has been trained (termed input space hereafter). This ensures that new learning processes will not interfere with the learned tasks

- [ICCV 2019] (RECOMMANDED!) ([paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Dvornik_Diversity_With_Cooperation_Ensemble_Methods_for_Few-Shot_Classification_ICCV_2019_paper.pdf)) Diversity with Cooperation: Ensemble Methods for Few-Shot Classification
    * New way to solve few-shot learning problems without meta-learing.


#### External Memory
- [ICML 2016] Meta-Learning with Memory-Augmented Neural Networks

    This work lead NTM into the image classification, technically, this work should not belong to the few-shot problems.
    This method can identify the image labels, even the true label of current image are inputed along with the next image.

- [CVPR 2018] ([paper](https://arxiv.org/pdf/1804.08281.pdf)) Memory Matching Networks for One-Shot Image Recognition
- [ICLR 2019] ([paper](https://arxiv.org/pdf/1902.02527.pdf) [code](https://github.com/cogentlabs/apl.)) Adaptive Posterior Learning-Few-Shot Learning with a Surprise-Based Memory Module

#### Architecture
- [ICML 2017] Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1805.07722.pdf)) Task-Agnostic Meta-Learning for Few-shot Learning

    A training method force model to learn a unbiased initial model without over-performing on some particular tasks.

#### Task Representation and Measure
- [ICCV 2019] ([paper](https://arxiv.org/pdf/1902.03545.pdf)) (RECOMMENDED!) TASK2VEC- Task Embedding for Meta-Learning
    * Use Fisher information matrix to judge which backbone is suitable for current task.

#### Multi Label Image Classification
- [CVPR 2019 oral] ([paper](https://arxiv.org/pdf/1902.09811.pdf)) LaSO-Label-Set Operations networks for multi-label few-shot learning

#### Add Additional Informations
- [ICCV 2019] ([paper](https://arxiv.org/pdf/1812.09213.pdf) [code](https://sites.google.com/view/comprepr/home)) Learning Compositional Representations for Few-Shot Recognition

    Add additional annotations to the classes.

- [CVPR 2019] ([paper](https://arxiv.org/pdf/1904.03472.pdf)) Few-shot Learning via Saliency-guided Hallucination of Samples

    Form segmentations and mix up, aiming at eliminates the back ground noise.

- [ICCV 2019] ([paper](https://arxiv.org/pdf/1906.05186.pdf)) Boosting Few-Shot Visual Learning with Self-Supervision

    Self-supervision means to rotate itself, and compute two losses.

#### Self-training

- [NIPS 2019] ([paper](https://arxiv.org/pdf/1906.00562.pdf)) Learning to Self-Train for Semi-Supervised Few-Shot Classification

    Label the query set for the first run, then retrain the model with the pesudo label for the second run. (Simple but effective)


## Results in Datasets
Basically, we use [Omniglot](https://github.com/brendenlake/omniglot), [mini-Imagenet](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning),
 [tiredImagenet](https://arxiv.org/abs/1803.00676), [CUB 2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and full [Imagenet](http://image-net.org) for the datasets. We list the latest methods' performs in mini-Imagenet.
Welcome contributes to expand the tables of results. 

基本上在小样本图像分类领域，主流的数据集为 Omniglot，mini-Imagenet，tired-Imagenet，CUB 和完整的 ImageNet。在这里我们总结了当前已有的方法在 mini-ImageNet 上的表现。
非常欢迎大家来补充呀。

#### [mini-Imagenet](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning)

| Years | Methods              | Backbone | 5-way 1-shot    | 5-way 5-shot    |
|-------|----------------------|----------|-----------------|-----------------|
| 2016  | Matching Network     | Conv4    | 43.56 +- 0.84%  | 55.31% +- 0.73% |
| 2017  | MAML                 | Conv4    | 48.7% +- 1.84%  | 63.15% +- 0.91% |
| 2017  | Prototypical Network | Conv4    | 49.42% +- 0.78% | 68.20% +- 0.66% |
| 2018  | Relation Network     | Conv4    | 50.44% +- 0.82% | 65.32% +- 0.70% |
| 2018  | MetaGAN: An Adversarial Approach to Few-Shot Learning    |  Conv4   | 46.13+-1.78% | 60.71+-0.89% |
| 2019  | Incremental Few-Shot Learning with Attention Attractor Networks    | ResNet-10    | 54.95+-0.30 | 63.04+-0.30 |
| 2019  | Adaptive Cross-Modal Few-shot Learning    | ResNet-12    | 65.30 ±0.49% | 78.10 ± 0.36% |
| 2019  | Cross Attention Network for Few-shot Classification    | Conv4    | 67.19 ± 0.55 | 80.64 ± 0.35 |
| 2019  | Learning to Self-Train for Semi-Supervised Few-Shot Classification   | ResNet-12    | 70.1 ± 1.9 | 78.7 ± 0.8 |
| 2019  | Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning    | Conv-64F    | 51.24±0.74% | 71.02±0.64% |
| 2019  | Few-Shot Learning with Localization in Realistic Settings    | ResNet-50    | 49.64±.31% | 69.45±.28% |
| 2019  | Baby steps towards few-shot learning with multiple semantics    | ResNet-12    | 67.2 ± 0.4% | 74.8 ± 0.3% |
| 2019  | Generating Classification Weights with GNN Denoising Autoencoders for Few-Shot Learning    | WRN-28-10    | 62.96+-0.15% | 78.85+-0.10% |
| 2019  | Spot and Learn: A Maximum-Entropy Patch Sampler for Few-Shot Image Classification    | Conv4   | 47.18+-0.83% | 66.41+-0.67% |
| 2019  | Meta-Transfer Learning for Few-Shot Learning    | ResNet-12    | 61.2+-1.8% | 75.5+-0.8% |
| 2019  | Dense Classification and Implanting for Few-Shot Learning    | ResNet-12    | 62.53+-0.19% | 78.95+-0.13% |
| 2019  | Edge-Labeling Graph Neural Network for Few-shot Learning   | Conv4  |  | 66.85% |
| 2019  | Finding Task-Relevant Features for Few-Shot Learning by Category Traversal    | COnv4    | 41.62% | 58.77% |
| 2019  | Few-shot Learning via Saliency-guided Hallucination of Samples    | ResNet-12    | 65.30 ±0.49% | 78.10 ± 0.36% |
| 2018  | Memory Matching Networks for One-Shot Image Recognition    | Conv4    | 53.37+-0.48% | 66.97+-0.35% |
| 2018  | Dynamic Few-Shot Visual Learning without Forgetting    | ResNet-12    | 55.45+-0.89% | 70.13+-0.68% |
| 2018  | Few-Shot Learning with Global Class Representations    | Conv4    | 53.21+-0.40% | 72.34+-0.32% |
| 2018  | LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning    | Conv4    | 69.13+-0.35% | 72.28+-0.68% |
| 2018  | TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning    | ResNet-12    | 61.65 ± 0.15% | 76.36 ± 0.10% |
| 2019  | META-LEARNING WITH LATENT EMBEDDING OPTIMIZATION    | WRN-28- 10    | 61.76+-0.08% | 77.59+-0.12% |
| 2019  | LEARNING TO PROPAGATE LABELS: TRANSDUCTIVE PROPAGATION NETWORK FOR FEW-SHOT LEARNING  | Conv4    | 55.51% | 69.86% |
| 2018  | META-LEARNING FOR SEMI-SUPERVISED FEW-SHOT CLASSIFICATION    |  Conv4   | 50.09+-0.45% | 64.59+-0.28% |
| 2018  | A SIMPLE NEURAL ATTENTIVE META-LEARNER    |  Conv4   | 55.71+-0.99% | 68.88+-0.92% |
| 2017  | OPTIMIZATION AS A MODEL FOR FEW-SHOT LEARNING    |  Conv4   | 43.44+-0.77% | 60.60+-0.71% |
| 2019  | Centroid Networks for Few-Shot Clustering and Unsupervised Few-Shot Classification    |  Conv4   |  | 62.6+-0.5% |
| 2019  | Infinite Mixture Prototypes for Few-Shot Learning   |  Conv4   | 49.6+-0.8% | 68.1+-0.8% |


# More Direction
### Object Detection
- [CVPR 2017] ([paper](https://arxiv.org/pdf/1612.02559.pdf) [code](https://github.com/rkwitt/GuidedAugmentation)) AGA - Attribute-Guided Augmentation
  - Using external depth and pose informations

- [AAAI 2018] ([paper](https://arxiv.org/abs/1803.01529)) LSTD: A Low-Shot Transfer Detector for Object Detection

- [CVPR 2019] ([paper](https://arxiv.org/abs/1806.04728)) RepMet-Representative-based Metric Learning for Classification and Few-shot Object Detection
  - Representative learning 

- [CVPR 2019] ([paper](https://arxiv.org/pdf/1903.09372.pdf)) Few-shot Adaptive Faster R-CNN  
  - Domain adaptation & Few-shot classification head

- [CVPR 2019] Feature Selective Anchor-Free Module for Single-Shot Object Detection

- [ICCV 2019] ([paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Rahman_Transductive_Learning_for_Zero-Shot_Object_Detection_ICCV_2019_paper.html))Transductive Learning for Zero-Shot Object Detection
  - Fixed Pseudo-label and Dynamic Pseudo-label

- [ICCV 2019] ([paper](https://arxiv.org/pdf/1812.01866)) Few-shot Object Detection via Feature Reweighting
  - “Feature Reweighting” as weights & useful loss function

- [NIPS 2019] ([papre](https://arxiv.org/abs/1911.12529))One-Shot Object Detection with Co-Attention and Co-Excitation
  - Non-local Operate & Squeeze and Excitation  

### Segementation
- [CVPR 2019] CANet- Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning
- [AAAI 2019] ([paper](https://www.researchgate.net/publication/335296764_Attention-Based_Multi-Context_Guiding_for_Few-Shot_Semantic_Segmentation)) Attention-based Multi-Context Guiding for Few-Shot Semantic Segmentation
    * Utilize the output of the different layers between query branch and support branch to gain more context informations.

- [arXiv 2019] AMP-Adaptive Masked Proxies for Few-Shot Segmentation
    * Not sure result in this area.
- [AAAI 2019] Unsupervised Meta-learning of Figure-Ground Segmentation via Imitating Visual Effects
    * Differetiate the background from images. 


### Generative Model
- [ICCV 2019] ([paper](https://arxiv.org/pdf/1905.01723)) Few-Shot Unsupervised Image-to-Image Translation
- [ICCV 2019 best] ([paper](https://arxiv.org/abs/1905.01164) [code](https://github.com/tamarott/SinGAN)) SinGAN: Learning a Generative Model from a Single Natural Image
- [CVPR 2018] Multi-Content GAN for Few-Shot Font Style Transfer
- [NIPS 2019] ( [paper](https://nvlabs.github.io/few-shot-vid2vid/main.pdf) [code](https://nvlabs.github.io/few-shot-vid2vid/) )Few-shot Video-to-Video Synthesis

### Domain Adaptation
- [NIPS 2017] Few-Shot Adversarial Domain Adaptation
- [ICCV 2019] Bidirectional One-Shot Unsupervised Domain Mapping

### Reinforcement Learning
- [ICML 2019] Few-Shot Intent Inference via Meta-Inverse Reinforcement Learning

### Visual Tracking
- [ICCV 2019] Deep Meta Learning for Real-Time Target-Aware Visual Tracking

### Others
- [IJCAI 2019] Incremental Few-Shot Learning for Pedestrian Attribute Recognition
- [AAAI 2018] AffinityNet- Semi-supervised Few-shot Learning for Disease Type Prediction
    * Use few-shot method to enhance oringal disease type prediction

- [arXiv 2019] ([paper](https://arxiv.org/pdf/1902.10482.pdf)) Few-Shot Text Classification with Induction Network
    * Introduce dynamic routing to generate better class representations. One real industrial project.

- [NIPS 2018] Neural Voice Cloning with a Few Samples
- [IJCAI 2019] Meta-Learning for Low-resource Natural Language Generation in Task-oriented Dialogue Systems
- [ICCV 2019] ([paper](https://arxiv.org/pdf/1909.01205)) Few-Shot Generalization for Single-Image 3D Reconstruction via Priors
- [ICCV 2019] ([paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_ACMM_Aligned_Cross-Modal_Memory_for_Few-Shot_Image_and_Sentence_Matching_ICCV_2019_paper.pdf)) ACMM: Aligned Cross-Modal Memory for Few-Shot Image and Sentence Matching
- [ICCV 2019] (RECOMMANDED!) Task-Driven Modular Networks for Zero-Shot Compositional Learning
    * An interesting usage of a bunch of MLPs.
- [CVPR 2019] ([paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Rostami_SAR_Image_Classification_Using_Few-Shot_Cross-Domain_Transfer_Learning_CVPRW_2019_paper.pdf) [code](https://github.com/MSiam/AdaptiveMaskedProxies.)) SAR Image Classification Using Few-shot Cross-domain Transfer Learning
- [AAAI 2019] Hybrid Attention-based Prototypical Networks for Noisy Few-Shot Relation Classification
    * Relation Classification with FewRel
- [AAAI 2019] Few-Shot Image and Sentence Matching via Gated Visual-Semantic Embedding
    * Image and Sentence Matching
- [AAAI 2018] Few Shot Transfer Learning BetweenWord Relatedness and Similarity Tasks Using A Gated Recurrent Siamese Network

# Other Awesome Resources
We collect some awesome code and blogs here.
(Note that if you are now writing a few-shot papers, feel free to checkout `resources` file to get some bib there)

### Relevant Awesome Datasets Repo
### Relevant Awesome Few-shot PlayGround Repo
### Relevant Awesome Blogs

# How to recommend a paper
You are highly welcome to recommend a paper to this repo. 
The only thing you need to do is make a new issue with its name, conference name, years and some recommends words(no more than 400 words).

非常欢迎大家来推荐相关论文呀，推荐论文的方式非常简单，只需要提交一个 Issue，并在 Issue 中写清楚论文的题目，发表的会议名称以及年份和一个不超过 400 字的推荐理由即可。

> EXAMPLE
>
> Title: [ICML 2019] TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning
>
> Recommend: First paper point out how to measure the backbone is bad or good for the current task(episode).

# Main Contributors
- [Duan-JM](www.github.com/Duan-JM) (Image Classification)

