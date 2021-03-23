## Image Classification
As far as I known, this file contains all papers in top conferences till now.
Feel free to let me know if there is more awesome paper in image
classification.

这个文件里面存放着至今为止，所有和小样本图像分类相关的论文。如果你有相关顶会论文想推荐的话，非常欢迎在
issue 里面提出来，我会在第一时间进行回复的。

### Contents
<!-- vim-markdown-toc GitLab -->

* [Surveys](#surveys)
* [Challenges](#challenges)
* [Theoretical Analysis](#theoretical-analysis)
* [Parameter Optimize Based Few-shot Learning](#parameter-optimize-based-few-shot-learning)
  * [Papers](#papers)
* [Generative Based Few-shot Learning](#generative-based-few-shot-learning)
  * [Papers](#papers-1)
* [Metric Based Few-shot Learning](#metric-based-few-shot-learning)
  * [Classic Methods](#classic-methods)
  * [Features Extractor Enhanced Methods](#features-extractor-enhanced-methods)
  * [Proto-Enhanced Methods](#proto-enhanced-methods)
  * [Metric Functions / Graph based methods](#metric-functions-graph-based-methods)
* [Special](#special)
  * [Unsorted](#unsorted)
  * [External Memory](#external-memory)
  * [Architecture](#architecture)
  * [Task Representation and Measure](#task-representation-and-measure)
  * [Multi Label Image Classification](#multi-label-image-classification)
  * [3D Image Classification](#3d-image-classification)
  * [Incremental Learning](#incremental-learning)
  * [Add Additional Informations (Cross-modal)](#add-additional-informations-cross-modal)
  * [Self-training](#self-training)
* [Results in Datasets](#results-in-datasets)
  * [mini-Imagenet](#mini-imagenet)

<!-- vim-markdown-toc -->
### Surveys
- [ACM Computing Surveys 2020] ([paper](https://arxiv.org/pdf/1904.05046.pdf)) Generalizing from a Few Examples A Survey on Few-Shot Learning
- [ICLR 2019] ([paper](https://arxiv.org/pdf/1904.04232) [code](https://github.com/wyharveychen/CloserLookFewShot)) A Closer Look At Few-shot Classification
- [arXiv 2020] A New Meta-Baseline for Few-Shot Learning
- [arXiv 2020] A Comprehensive Overview and Survey of Recent Advances in Meta-Learning
- [arXiv 2020] Defining Benchmarks for Continual Few-Shot Learning
- [ICML 2020] Unraveling Meta-Learning: Understanding Feature Representations for Few-Shot Tasks
    * Choice to impact of meta-learning methods, and design module over it
- [arXiv 2020] ([paper](https://arxiv.org/pdf/2009.02653.pdf)) Learning from very few samples: A survey

### Challenges
- [ICLR 2020] ([paper](https://arxiv.org/pdf/1909.02729.pdf)) A Baseline for Few-shot Image Classification
- [ECCV 2020] ([code](http://github.com/WangYueFt/rfs/)) Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?
    * Similar to ICLR 2020 above, they challenge that simple finetune can be
        better than sophisticated meta-learning algorithms.
    * Used Technique are NN / LR / L-2 / Aug / Distill
    * With all techniques results are 62.02% or 64.82% 1-shot and 79.64% or 82.14% 5-shot on mini-Imagenet
- [ECCV 2020] A Broader Study of Cross-Domain Few-Shot Learning
    * This paper challenge few-shot learning methods by proposing a new benchmark
    * BTW, I not agree with one conclusion that earlier meta-learning methods perform better than latest methods. MetaOptNet indeed may under perform with PN, but I prefer to make compassion with TADAM or simple CNAPS.

### Theoretical Analysis

- [NIPS 2018 Bengio] ([paper](https://papers.nips.cc/paper/7504-metagan-an-adversarial-approach-to-few-shot-learning.pdf)) MetaGAN An Adversarial Approach to Few-Shot Learning

- [ICLR 2020] A THEORETICAL ANALYSIS OF THE NUMBER OF SHOTS IN FEW-SHOT LEARNING
    * Analysis on Prototypical Networks, result is not significant, however we can have a look into the analysis
    * Analysis the phenomena that when training shot mismatch the val shot, the prototype's perform will degenerate.
    * The review score is 8, 6, 6. I am little argue with the result, for train on small shot then evaluate on large shot could improve the result.  Furthermore, only 0.X% improvement or versus could casued by some other reasons, such like random seed, initializations, so I am little double with the results.

- [ICLR 2020] Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML
    * Explore MAML then propose ANIL

- [ICML 2020] On the Global Optimality of Model-Agnostic Meta-Learning
    * Theoretical analysis for meta-learning

- [ICML 2020] Meta-learning for mixed linear regression
    * Theoretical analysis on how many auxiliary tasks can help small-data task

- [ICML 2020] A Sample Complexity Separation between Non-Convex and Convex Meta-Learning
    * RECOMMENDED!

- [ECCV 2020 spotlight] ([paper](https://arxiv.org/abs/2003.12060) [code](https://github.com/bl0/negative-margin.few-shot)) Negative Margin Matters: Understanding Margin in Few-shot Classification
    * [Author @bl0]Introduces a negative margin loss to metric learning based few-shot learning methods and achieves state-of-the-art accuracy. To understand why the negative margin loss performs well, we provide both the intuitive and theoretical analysis.

- [ECCV 2020] Impact of base dataset design on few-shot image classification

- [ECCV 2020] When Does Self-supervision Improve Few-shot Learning?

- [NIPS 2020] Interventional Few-Shot Learning (Need to be cautious)
    * First paper to try theory in the book of why in pratical
    * However, maybe still miss some assumptions
    * Recommand people who interested in this paper to read 7th chapter of the book of why.(Focus on the the three rules for do operation)

- [ICLR 2021] ([paper](https://openreview.net/pdf?id=pW2Q2xLwIMD)) FEW-SHOT LEARNING VIA LEARNING THE REPRESENTATION, PROVABLY
    * TL;DR: demonstrate the advantage of representation learning in both high-dimensional linear regression and neural networks, and show that representation learning can fully utilize all n_1T samples from source tasks.

### Parameter Optimize Based Few-shot Learning
** Descriptions 01**: They use normal classifier to classify samples
** Descriptions 02**: They mainly generate/optimize classifier parameters

#### Papers
** Descriptions 01**: TODO: We will sort following papers later

- [ICML 2017] Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
- [ICLR 2017 Meta-learner LSTM Ravi] ([paper](https://openreview.net/pdf?id=rJY0-Kcll&source=post_page---------------------------) [code](https://github.com/twitter/meta-learning-lstm.)) Optimization as a Model for Few-shot Learning
    * Use LSTM to generate classifier's parameters

- [NIPS 2018] Probabilistic model-agnostic meta-learning
- [NIPS 2018] Bayesian model-agnostic meta-learning
- [arXiv 2018 REPTILE] ([paper](https://arxiv.org/pdf/1803.02999.pdf)) On First-Order Meta-Learning Algorithms
- [ICLR 2018 SNAIL] ([paper](https://arxiv.org/pdf/1707.03141.pdf)) A Simple Neural Attentive Meta- Learner
    * Improve the Meta-Learner LSTM, by adding temporal convolution and caual attention to the network.

- [CVPR 2018] ([paper](https://arxiv.org/pdf/1804.09458.pdf) [code](https://github.com/gidariss/FewShotWithoutForgetting)) Dynamic Few-Shot Visual Learning without Forgetting
- [CVPR 2018] ([paper](https://arxiv.org/pdf/1712.07136.pdf)) Low-Shot Learning with Imprinted Weights
    * Passing, generate weights for classifier. (I DONOT like it, the exp only compare to matching networks and "Generative + classifier")

- [NIPS 2018] Delta-encoder: an effective sample synthesis method for few-shot object recognition
- [CVPR 2018] Few-Shot Image Recognition by Predicting Parameters From Activations

- [CVPR 2019] (RECOMMENDED!) ([paper](https://arxiv.org/pdf/1710.06177.pdf)) Learning to Learn Image Classifiers with Visual Analogy
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1903.05050.pdf)) Dense Classification and Implanting for Few-Shot Learning
- [ICML 2019] ([paper](https://arxiv.org/pdf/1905.06331.pdf) [code](https://github.com/likesiwell/LGM-Net/)) LGM-Net: Learning to Generate Matching Networks for Few shot Learning
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1904.05967.pdf) [code](https://github.com/ucbdrive/tafe-net)) TAFE-Net- Task-Aware Feature Embeddings for Low Shot Learning
    * Use a meta-learner to generate parameters for the feature extractor


- [ICLR 2019 LEO Vinyals] (RECOMMENDED!) ([paper](https://arxiv.org/pdf/1807.05960.pdf) [code](https://github.com/deepmind/leo)) Meta-learning with latent embedding optimization
    * High dimensional problem is hard to solve in the low-data circumstances, so this work try to bypass the limitations by learning a data-dependent latent low-dimensional latent space of model parameters.

- [ICLR 2019] CAML: FAST CONTEXT ADAPTATION VIA META-LEARNING
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1905.01102.pdf) [code](https://github.com/gidariss/wDAE_GNN_FewShot)) Generating Classification Weights with GNN Denoising Autoencoders for Few-Shot Learning
    * Little better than LEO

- [ICLR 2019] Meta-learning with differentiable closed-form solvers
    * Teach to use traditional machining learning methods
    * Most likely no good than LEO

- [ICLR 2020] META-LEARNING WITH WARPED GRADIENT DESCENT
    * feel not outperform the LEO

- [ICLR 2020] LEARNING TO BALANCE: BAYESIAN META-LEARNING FOR IMBALANCED AND OUT-OF-DISTRIBUTION TASKS
    * try to solve task- and class-imbalanced problems
    * feel not significant on normal few-shot learning setup

- [CVPR 2019] Task Agnostic Meta-Learning for Few-Shot Learning
    * Improve MAML to 66%
    * the initial model can be trained biased towards some tasks, particularly those sampled in meta-training phase

- [NIPS 2019] ([paper](https://arxiv.org/abs/1910.13616) [code](https://github.com/shaohua0116/MMAML-Classification)) Multimodal Model-Agnostic Meta-Learning via Task-Aware Modulation
    * Modal in this paper is not mean visual or semantic information, but refer
        to different task distributions. For example, Omniglot and mini-Imagenet are regaled as two mode.
    * They augment MAML with the capability to identify the mode of tasks sampled from a multi-modal task distribution and adapt quickly through gradient updates. 
    * Architecture is similar to TADAM, just check it out

- [ICLR 2020] Automated Relational Meta-learning
    * adding knowledge graph on the prototypes
    * not familiar with the dataset, but very interesting
    * show 5% better than TADAM
    * nearly no effect on mini-Imagenet (have room for improvement)

- [ICLR 2020] ES-MAML: Simple Hessian-Free Meta Learning
    * estimate second derivatives using bp is difficult, ES avoids the problem of estimating second derivatives

- [ICML 2020] Learning to Stop While Learning to Predict
    * Plug an stop mechanism onto the MAML to avoid "over-thinking"

- [ICLR 2020] ([code](https://github.com/amzn/xfer)) Empirical Bayes Transductive Meta-Learning with Synthetic Gradients
    * semi-supervised learning, using model to synthetic fake gradients to
        simulate the true gradients on query set
    * WRN-28-10(pre-trained) 70% for 1-shot and 79 for 5-shot
    * abbrev is SIB in other papers

- [ECCV 2020] ([code]( https://gitlab.mpi-klsb.mpg.de/yaoyaoliu/e3bm))An Ensemble of Epoch-wise Empirical Bayes for Few-shot Learning
    * addon on the SIB(semi) 71.4% for 1-shot and 81.2% for 5-shot in miniImagnet

- [ICML 2020] MetaFun: Meta-Learning with Iterative Functional Updates
- [ECCV 2020 spotlight]  Associative Alignment for Few-shot Image Classification
    * This paper proposes the idea of associative alignment for leveraging part of the base data by aligning the novel training instances to the closely related ones in the base training set.
    * WRN-28-10: 65.92% for 1-shot and 82.85% for 5-shot on mini-ImageNet

- [CVPR 2020] TransMatch: A Transfer-Learning Scheme for Semi-Supervised Few-Shot Learning
    * Imprinted weights to finetune 
    * SSL settings
    * 63% 1-shot 82% 5shot

- [ICLR 2021] ([paper](https://arxiv.org/pdf/2007.10417.pdf)) BAYESIAN FEW-SHOT CLASSIFICATION WITH ONE-VS-EACH PO ́ LYA-GAMMA AUGMENTED GAUSSIAN PROCESSES
    * They propose a Gaussian process classifier based on a novel combination of Po ́lya-Gamma augmentation and the one-vs-each softmax approximation (Titsias, 2016) that allows us to efficiently marginalize over functions rather than model parameters.

- [ICLR 2021] ([paper](https://arxiv.org/pdf/2008.08882.pdf)) BOIL: TOWARDS REPRESENTATION CHANGE FOR FEW-SHOT LEARNING
    * They investigate the necessity of representation change for the ultimate goal of few-shot learning, which is solving domain-agnostic tasks.
    * Only update body of extractor in the inner loop, to enhance MAML
    * TL;DR: The results imply that representation change in gradient-based meta-learning approaches is a critical component.

### Generative Based Few-shot Learning
** Descriptions 01**: To extend small datasets to larger one, they generate fake samples.

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
- [ECCV 2020 oral] ([paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460579.pdf))  Model-Agnostic Boundary-Adversarial Sampling for Test-Time Generalization in Few-Shot learning
- [ICLR 2021 oral] ([paper](https://arxiv.org/pdf/2101.06395.pdf)) FREE LUNCH FOR FEW-SHOT LEARNING: DISTRIBUTION CALIBRATION
    * Calibrate the distribution of these few-sample classes by transferring statistics from the classes with sufficient examples, then generate an adequate number of examples
    * Assume every dimension in the feature representation follows a Gaussian distribution
    * 68.57 ± 0.55 / 82.88 ± 0.42 on miniImagnet
    * 78.19 ± 0.25 / 89.90 ± 0.41 on tieredImagnet
    * 79.56 ± 0.87 / 90.67 ± 0.35 on CUB
- [ICLR 2021] ([paper](https://arxiv.org/pdf/2008.06981.pdf)) BOWTIE NETWORKS: GENERATIVE MODELING FOR JOINT FEW-SHOT RECOGNITION AND NOVEL-VIEW SYNTHESIS

### Metric Based Few-shot Learning
** Descriptions 01:** They classify target samples by using metric functions which are used to measure similarity between samples or classes.

#### Classic Methods
** Descriptions 01**: They are the foundations of the metric-based methods.
- [ICML 2012] One-Shot Learning with a Hierarchical Nonparametric Bayesian Model
    * Using Hierarchical Bayesian Model after extracted features. Which is similar to build the category graph method in IJCAI 2019.

- [ICML 2015] Siamese Neural Networks for One-Shot Image Recognition
- [NIPS 2016] Matching Networks for One Shot Learning
    * This methods cast the problem of one-shot learning within the set-to-set framework

- [NIPS 2017]  (RECOMMENDED!) Prototypical Networks for Few-shot Learning
- [CVPR 2018] Learning to Compare：Relation Network for Few-Shot Learning
    * Change metric functions to CNNs
    * Provide a clean framework that elegantly encompasses both few and zero-shot learning.


#### Features Extractor Enhanced Methods
** Descriptions 01**: They improve classic methods by enhancing feature extractors

- [NIPS 2018  TADAM] (RECOMMENDED!) TADAM-Task dependent adaptive metric for improved few-shot learning 
    * In every task, use task representations to finetune the output of each Conv Blocks, like BN functionally.

- [ECCV 2018] ([paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Fang_Zhao_Dynamic_Conditional_Networks_ECCV_2018_paper.pdf)) Dynamic Conditional Networks for FewShot Learning
    * Basically same as TADAM(using task representation to finetune the backbones), it use a conditional feature to influence the output of conv layers (Linear combination).

- [NIPS 2019 CNAPS] ([paper](https://arxiv.org/pdf/1906.07697.pdf) [code](https://github.com/cambridge-mlg/cnaps)) Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes
    * Similar to TADAM, both succeed from Film
    * gamma and beta are generated with task embeddings and images
    * Task embeddings are generated by separate encoders

- [CVPR 2020 simple CNAPS] ([paper](https://arxiv.org/abs/1912.03432) [code](https://github.com/peymanbateni/simple-cnaps))Improved Few-Shot Visual Classification
    * Belong to TADAM
    * Same very impressive result on mini-Imagenet
    * Change Distance in CNAPS

-  [ECCV 2020] Selecting Relevant Features from a Universal Representation for Few-shot Learning
    * mix-up the final feature with multi model output or multi depth layer in the models output
    * Specific configuration to each type of datasets and adapt to each of them

- [ICML 2020] TaskNorm: Rethinking Batch Normalization for Meta-Learning
    * We can compare with TADAM CNAPS simple CNAPS  - very interesting

- [CVPR 2019] ([paper](https://arxiv.org/pdf/1905.11116.pdf) [code](https://github.com/Clarifai/few-shot-ctm.)) Finding Task-Relevant Features for Few-Shot Learning by Category Traversal
    * Need attention to its code
    * Not recommended

- [AAAI 2019] A Dual Attention Network with Semantic Embedding for Few-shot Learning
    * Add spatial attention and task attention.

- [ICCV 2019] PARN Position-Aware Relation Networks for Few-Shot Learning
    * Improve on Relation Network, change a way to extract more information during feature extraction stage, then argue that when objects in the same class appear on the different spatial position will cause the misclassification, they redesign the metric module(the origianl 2CNN + 2FC) instead.  
    * Conv-4 71% on mini-imagenet 5shot

- [ECCV 2020] ([paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730239.pdf)) Attentive Prototype Few-shot Learning with Capsule Network-based Embedding
    * Use capsule as its feature extractor
    * 66%/82% for mini-Imagenet, 69%/86% for tiered-Imagenet 
    * I think it can proof that capsule's efficient for low-resource circumstances.

- [ICLR 2021] ([paper](https://openreview.net/pdf?id=vujTf_I8Kmc)) CONSTELLATION NETS FOR FEW-SHOT LEARNING
    * constellation block right behind each conv block
    * performs cell feature clustering and encoding with a dense part representation
    * captures the relationships among the cell features are further modeled by an attention mechanism

#### Proto-Enhanced Methods
** Descriptions 01**: They improve classic methods by enrich prototypes
** ATTENTION 01**: Proto-enhanced Cross-modal methods are not include here, found them below

- [ICLR 2018 Ravi] ([paper](https://arxiv.org/pdf/1803.00676.pdf) [code](https://github.com/renmengye/few-shot-ssl-public)) Meta-Learning for Semi-Supervised Few-Shot Classification
    * Using soft K-means to refine the prototypes, then using variant ways(training methods) to eliminate the outline points.
    * Create new datasets: tiered-Imagenet
- [CVPR 2019 oral] ([code](https://github.com/kjunelee/MetaOptNet)) Meta-Learning With Differentiable Convex Optimization
    * 5-shot 5-way 78% on mini-imagenet
- [ICML 2019] (RECOMMENDED!) ([paper](https://arxiv.org/pdf/1905.06549.pdf)) TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning
- [ICCV 2019] ([paper](https://arxiv.org/pdf/1908.05257)) Few-Shot Learning with Global Class Representations
    * Synthesis new samples to alleviate the data imbalance problem between Base and Novel Classes.
    * During training, compute two losses, one is the original losses, the other is the score for the whole classes including noval classes.
- [IJCAI 2019] ([paper](https://arxiv.org/pdf/1905.04042) [code](https://github.com/liulu112601/PPN)) Prototype Propagation Networks (PPN) for Weakly-supervised Few-shot Learning on Category Graph
    * Manually build an category graph, then add parents label's class representation into the child class representations.
- [CVPR 2019] (RECOMMENDED) ([paper](https://arxiv.org/pdf/1904.08482.pdf) [code](https://github.com/mibastro/VPE)) Variational Prototyping-Encoder- One-Shot Learning with Prototypical Images
    * Use encoder to translate the real images to abstract prototypes, such as painted traffic signs, then compare query and sample in the prototypes latent space.
- [CVPR 2019] ([paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chu_Spot_and_Learn_A_Maximum-Entropy_Patch_Sampler_for_Few-Shot_Image_CVPR_2019_paper.pdf)) Spot and Learn A Maximum-Entropy Patch Sampler for Few-Shot Image Classification
    * Sample parts of the image to form the batch to represent the class.
    * One-shot not pretty good(51%)
- [ECCV 2020 oral] Prototype Rectification for Few-Shot Learning
    * intra-cross bias -> soft-KNN methods to adjust proto
    * cross-class bias -> proposed a normalize term to force query set to support set
    * !!! IT HAS SOME THEORETICAL ANALYSIS !!!
    * Result impressive
- [NIPS 2019] Meta-Reinforced Synthetic Data for One-Shot Fine-Grained Visual Recognition
    * Use a Generator to generate fused Image to extend prototypes
- [ICML 2019] ([paper](https://arxiv.org/pdf/1902.04552.pdf)) (RECOMMENDED!) Infinite Mixture Prototypes for Few-shot Learning
    * Point out that data distribution for one class are not uni-model (Verify in my experiments too).
    * (Clustering methods) Semi-Supervised methods for prototypical networks. Show this methods even suit for unsupervised situations(protentially).
    * Improve on Alphabets dataset, remain or improve on omniglot and mini-imagenet.
- [ECCV 2020] TAFSSL: Task-Adaptive Feature Sub-Space Learning for few-shot classification
    * 77% for 1shot and 84.99 for 5-shot on mini-Imagenet
    * Motivation: Assume there exists "noizo" pattern contained in features
        extracted by backbone. They need to pick out which of them is useful for
        current task
    * Methods divided into two step: Dimensionality reduction(PCA or ICA) -> Cluster methods(BKM or MSP)
- [ICML 2020] Meta Variance Transfer: Learning to Augment from the Others
    * Use support categories to enhance target categories embeddings
- [ICML 2020] Meta-Learning with Shared Amortized Variational Inference
    * Using Shared Amortized Variational Inference to generalize more precise prototypes
- [CVPR 2020] Adaptive Subspaces for Few-Shot Learning
- [ICLR 2021] ([paper](https://openreview.net/pdf?id=D3PcGLdMx0)) MELR: META-LEARNING VIA MODELING EPISODE- LEVEL RELATIONSHIPS FOR FEW-SHOT LEARNING
    * Build up attention model between adjacency episode
    * ResNet12 - 67.40% / 83.40% for mini-ImageNet 72.14%/87.01% for tiered-ImageNet

**cross-modal**:
  You can find relevant cross-modality methods below

#### Metric Functions / Graph based methods 

- [CVPR 2018] (RECOMMENDED!) Low-Shot Learning With Large-Scale Diffusion
- [ICLR 2018] Few-Shot Learning with Graph Neural Networks
- [NIPS 2019] Learning to Propagate for Graph Meta-Learning
    * Learns to propagate messages between prototypes of different classes on the graph, so that learning the prototype of each class benefits from the data of other related classes.
    * Attention mechanic.
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1905.01436.pdf)) Edge-Labeling Graph Neural Network for Few-shot Learning
- [ICLR 2019] ([paper](https://arxiv.org/pdf/1805.10002.pdf)) Learning to Propagate Labels-transductive Propagation Network for Few-shot Learning
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1903.12290.pdf) [code](https://github.com/WenbinLee/DN4.git)) Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning
    * Calculating the similarity between query and class represent feature in feature level, rather than instance level. It seperate original feature in m part and then compute the similarity to the K-nearst class partial features.
    * Good Result on mini-ImageNet 71.02 ± 0.64% for 5shot and 51.24% for 1shot with Conv4_64F.
- [ICCV 2019] Transductive Episodic-Wise Adaptive Metric for Few-Shot Learning
- [AAAI 2019] Distribution Consistency based Covariance Metric Networks for Few Shot Learning
    * Slight improve on 1-shot compare to Relation Network, however degenerate on 5-shot compare to Protoypical Network.
- [ACMMM 2019] TGG: Transferable Graph Generation for Zero-shot and Few-shot Learning
- [ICCV 2019] ([paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Hao_Collect_and_Select_Semantic_Alignment_Metric_Learning_for_Few-Shot_Learning_ICCV_2019_paper.pdf)) Collect and Select: Semantic Alignment Metric Learning for Few-Shot Learning
    * Use attention to pick(Select) most relevant part to compare
- [CVPR 2020] ([code](https://github.com/megvii-research/DPGN) DPGN: Distribution Propagation Graph Network for Few-shot Learning
    * 67% for 1-shot 84% for 5-shot (ResNet18), however Conv can achieve 66.01%
    * Construct two graph, one for sample-wise one for distribution
- [CVPR 2020 oral] ([code](https://github.com/icoz69/DeepEMD)) DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover's Distance and Structured Classifiers
    * 65.91 1-shot， 82.41 5-shot
    * new metric functions
- [AAAI 2020]  Knowledge Graph Transfer Network for Few-Shot Recognition
    * Knowledge Graph Transfer Network for Few-Shot Recognition
    * did not compare in mini-imagenet, seems fine
- [AAAI 2020] Variational Metric Scaling for Metric-Based Meta-Learning
    * 60% for 1-shot on TADAM and 77% for 5-shot on mini-imagenet
- [ICLR 2020] FEW-SHOT LEARNING ON GRAPHS VIA SUPER-CLASSES BASED ON GRAPH SPECTRAL MEASURES
- [ICML 2020] ([paper](https://arxiv.org/pdf/2002.02050.pdf) [exist code](https://github.com/JiechaoGuan/FSL-DAPNA)) Few-Shot Learning as Domain Adaptation: Algorithm and Analysis
    * Using MDD loss, which is very common in domain adaptation areas, to align features between episodes
    * 71.88% for 1-shot and 84.07 for 5-shot with WRN-28-10 mini-ImageNet
    * 69.14% for 1-shot and 85.82 for 5-shot on tiered-ImageNet
- [CVPR 2020] [ [exist code](https://github.com/Sha-Lab/FEAT.) ] Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions
    * 66% for 1-shot 82 for 5-shot
- [ECCV 2020] ([code](https://github.com/ElementAI/embedding-propagation)) Embedding Propagation: Smoother Manifold for Few-Shot Classification
    * ElementAI
    * 70 1-shot 81 5-shot
    * Using rotation as self-supervised during pre-trained period
    * Perform label propagation during finetune period
    * [WRN-28-10] 70.74 for 1-shot and 84.34% for 5-shot for mini-imagenet
    * [WRN-28-10] 79.22 for 1-shot and 88.05% for 5-shot under SSL settings for mini-imagenet
- [ECCV 2020] SEN: A Novel Dissimilarity Measure for Prototypical Few-Shot Learning Networks
- [AAAI 2021] ([paper](https://arxiv.org/abs/2012.10844)) PTN: A Poisson Transfer Network for Semi-supervised Few-shot Learning
    * 82.66% for 1shot and 88.43% for 5shot on mini-Imagenet with WRN-28-10
    * 84.70% for 1shot and 89.14% for 5shot on tiered-Imagenet with WRN-28-10
- [ICLR 2021] ([paper](https://openreview.net/pdf?id=eJIJF3-LoZO)) CONCEPT LEARNERS FOR FEW-SHOT LEARNING
    * Seperate single prototype to several prototypes which represents one
        concept each
    * Comparaed on CUB / Tabula Muris / Reuters

### Special
#### Unsorted
- [Nature 子刊 MI 2018] ([paper](https://arxiv.org/pdf/1810.01256.pdf)) Continuous Learning of Context-dependent Processing in Neural Networks
    * During training a network consecutively for different tasks, OWNs weights are only allowed to be modified in the direction orthogonal to the subspace spanned by all inputs on which the network has been trained (termed input space hereafter). This ensures that new learning processes will not interfere with the learned tasks

- [ICCV 2019] (RECOMMANDED!) ([paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Dvornik_Diversity_With_Cooperation_Ensemble_Methods_for_Few-Shot_Classification_ICCV_2019_paper.pdf)) Diversity with Cooperation: Ensemble Methods for Few-Shot Classification
    * New way to solve few-shot learning problems without meta-learing.
    * Like title, accuracy of 81% on 5-shot mini-imagenet

- [ICCV 2019] Variational Few-Shot Learning
- [NIPS 2017] Few-Shot Learning Through an Information Retrieval Lens
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1904.08502)) Few-Shot Learning with Localization in Realistic Settings
    * Locate the object in the images first, then classify them.
    * Classify in real-world images, somehow not interesting.
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1812.02391v2.pdf)) Meta-Transfer Learning for Few-Shot Learning
    * nearly no improve on 5 way 5 shot on mini-ImageNet.
    * In ECCV 2020 paper, with ResNet-25 can achieve 63.4 for 1-shot and 81.17 for 5-shot on mini-ImageNet
- [ICCV 2019] Few-Shot Learning with Embedded Class Models and Shot-Free Meta Training
    * Perform well in 5-5 train to 5-1 test
    * 77% for miniimagenet 5-5shot
- [AAAI 2020] [exist code] Diversity Transfer Network for Few-Shot Learning
    * 63% for 1-shot 77.9% for 5-shot
    * using external reference image to form auxiliary tasks
- [CVPR 2020] Adversarial Feature Hallucination Networks for Few-Shot Learning
    * Two novel regularizes: classification regularize and anti-collapse regularize
    * 62.38% 1-shot 78.16 5-shot
- [CVPR 2020] Revisiting Pose-Normalization for Fine-Grained Few-Shot Recognition
- [ICLR 2020] META DROPOUT: LEARNING TO PERTURB LATENT FEATURES FOR GENERALIZATION
- [ICML 2020] ([paper](https://arxiv.org/abs/2006.15486)) Laplacian Regularized Few-Shot Learning
    * Regulation term
        * a unary term assigning query samples to the nearest class prototype
        * a pairwise Laplacian term encouraging nearby query samples to have consistent label assignments
    * No re-train the base model: graph clustering of the query set, subject to
        supervision constraints from the support set
    * ResNet-18 72.11 for 1-shot and 82.31 for 5-shot
    * WRN 74.86 for 1-shot and 84.13 for 5-shot
- [ICML 2020] Informative Dropout for Robust Representation Learning: A Shape-bias Perspective
- [NIPS 2020] Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels
- [CVPR 2020] Meta-Learning of Neural Architectures for Few-Shot Learning
- [NIPS 2020] Transductive Information Maximization for Few-Shot Learning
    * TIM-GD 77.8% for 1-shot and 87.4% for 5-shot for mini-Imagenet
- [ICLR 2021] METANORM: LEARNING TO NORMALIZE FEW-SHOT BATCHES ACROSS DOMAINS
    * add norm over meta-tasks
- [ICLR 2021] ([paper](https://openreview.net/pdf?id=xzqLpqRzxLq)) IEPT: INSTANCE-LEVEL AND EPISODE-LEVEL PRE- TEXT TASKS FOR FEW-SHOT LEARNING
    * Add three addition loss (instance-level SSL, cross episode SSL)
    * 67.05% / 82.90% mini-ImageNet, 72.24% / 86.73% tiered-ImageNet
- [AISTATS 2021] ([paper](https://arxiv.org/abs/2102.00127)) On Data Efficiency of Meta-learning
    * AISTATS 2021 is Rank A conference
    * Compare methods on ImageNet Omniglot EMNIST
- [AAAI 2021] ([paper](https://arxiv.org/pdf/2009.04724.pdf)) Attributes-Guided and Pure-Visual Attention Alignment for Few-Shot Recognition

#### External Memory
- [ICML 2016] Meta-Learning with Memory-Augmented Neural Networks

    This work lead NTM into the image classification, technically, this work should not belong to the few-shot problems.
    This method can identify the image labels, even the true label of current image are inputed along with the next image.

- [CVPR 2016] Few-Shot Object Recognition From Machine-Labeled Web Images

- [CVPR 2018] ([paper](https://arxiv.org/pdf/1804.08281.pdf)) Memory Matching Networks for One-Shot Image Recognition
- [ICLR 2019] ([paper](https://arxiv.org/pdf/1902.02527.pdf) [code](https://github.com/cogentlabs/apl)) Adaptive Posterior Learning-Few-Shot Learning with a Surprise-Based Memory Module
- [CVPR 2020] ([paper](https://github.com/Yikai-Wang/ICI-FSL)) Instance Credibility Inference for Few-Shot Learning
    * pick out most trust unlabeled samples, then re train classifier
    * 71% for 1-shot and 81% for 5shot
- [ICCV 2019] Few-Shot Learning with Embedded Class Models and Shot-Free Meta Training
    * life-long few-shot learning problems
- [ICLR 2020] Meta-Learning Deep Energy-Based Memory Models
- [TKDE 2020] Many-Class Few-Shot Learning on Multi-Granularity Class Hierarchy
- [NIPS 2020] Learning to Learn Variational Semantic Memory

#### Architecture
- [CVPR 2019] ([paper](https://arxiv.org/pdf/1805.07722.pdf)) Task-Agnostic Meta-Learning for Few-shot Learning
    * A training method force model to learn a unbiased initial model without over-performing on some particular tasks.

- [ECCV 2020] Meta-Learning across Meta-Tasks for Few-Shot Learning
    * Build Cross-Domain methods between meta-tasks
    * WRN 72.41% 1-shot 84.34% 5-shot for mini-Imagenet
    * Build MKD and MDA across tasks to make model to learn relationship between same tasks and different tasks

#### Task Representation and Measure
- [ICCV 2019] ([paper](https://arxiv.org/pdf/1902.03545.pdf)) (RECOMMENDED!) TASK2VEC- Task Embedding for Meta-Learning
    * Use Fisher information matrix to judge which backbone is suitable for current task.

#### Multi Label Image Classification
- [CVPR 2019 oral] ([paper](https://arxiv.org/pdf/1902.09811.pdf)) LaSO-Label-Set Operations networks for multi-label few-shot learning
- [TPAMI 2020] Knowledge-Guided Multi-Label Few-Shot Learning for General Image Recognition

#### 3D Image Classification
- [ICLR 2021] ([paper](https://openreview.net/pdf?id=-Lr-u0b42he)) DISENTANGLING 3D PROTOTYPICAL NETWORKS FOR FEW-SHOT CONCEPT LEARNING
    * 3D image classification

#### Incremental Learning
- [CVPR 2020 oral] ([paper](https://arxiv.org/pdf/2004.10956.pdf)) Few-Shot Class-Incremental Learning
    * class-incremental problems
    * continue-learning
- [ICML 2020] XtarNet: Learning to Extract Task-Adaptive Representation for Incremental Few-Shot Learning
    * Task specific generate classifier weights
    * Task specific Meta-CNN branch
- [NIPS 2019] ([paper](https://arxiv.org/pdf/1810.07218.pdf) [code](https://github.com/renmengye/inc-few-shot-attractor-public)) Incremental Few-Shot Learning with Attention Attractor Networks
    * Using normal way to pretrain the backbone on the base classes, then using the base class weights to fintune the classifier on the few-shot episodic network.
    * Achieve the normal
- [ECCV 2020] Incremental Few-Shot Meta-Learning via Indirect Feature Alignment
- [ICLR 2021] ([paper](https://openreview.net/pdf?id=3SV-ZePhnZM)) INCREMENTAL FEW-SHOT LEARNING VIA VECTOR QUANTIZATION IN DEEP EMBEDDED SPACE
- [ICLR 2021] ([paper](https://openreview.net/pdf?id=oZIvHV04XgC)) WANDERING WITHIN A WORLD: ONLINE CONTEXTUALIZED FEW-SHOT LEARNING
    * Continue Learning
    * New Datasets

#### Add Additional Informations (Cross-modal)
- [NIPS 2019] ([paper](https://arxiv.org/pdf/1910.07677.pdf)) Cross Attention Network for Few-shot Classification
    * Learn a attention(mask) to pay more attention on the part of the images
    * Add transductive inference part
    * Pretty good result on mini-imagenet 80.64 +- 0.35% under ResNet-12 (16 conv layers)

- [NIPS 2019] ([paper](https://arxiv.org/pdf/1902.07104.pdf)) Adaptive Cross-Modal Few-shot Learning
    * Using texture information to enhance the performance, which reach a comparable result on mini-imagenet
    * Perform well on 1-shot rather than 5-shot or 10-shot

- [CVPR 2019] ([paper](https://arxiv.org/pdf/1906.01905.pdf)) Baby steps towards few-shot learning with multiple semantics
    * Show 4.5 years old baby perform 70% on 1-shot case, adult achieve 99%.
    * Add multi-semantic into the task.
    * However on 5-shot case LEO perform exceed both this paper and the paper above with no semantics information.
    * For 1-shot case, this method achieve 67.2% +- 0.4% compare to 70% of human baby performance.

- [ICCV 2019] ([paper](https://arxiv.org/pdf/1812.09213.pdf) [code](https://sites.google.com/view/comprepr/home)) Learning Compositional Representations for Few-Shot Recognition
    Add additional annotations to the classes.

- [CVPR 2019] ([paper](https://arxiv.org/pdf/1904.03472.pdf)) Few-shot Learning via Saliency-guided Hallucination of Samples
    Form segmentations and mix up, aiming at eliminates the back ground noise.

- [ICCV 2019] ([paper](https://arxiv.org/pdf/1906.05186.pdf)) Boosting Few-Shot Visual Learning with Self-Supervision
    * Self-supervision means to rotate itself, and compute two losses.

- [ICCV 2019] ([paper](https://pdfs.semanticscholar.org/9d04/7a9c96d1e929846b28a44498a230fffee06f.pdf?_ga=2.165168639.132909448.1580616762-480481026.1580441958)) Few-Shot Image Recognition With Knowledge Transfer

- [CVPR 2019] Large-Scale Few-Shot Learning- Knowledge Transfer With Class Hierarchy
    * Aiming at learning large-scale problem, not just on 5 novel class.
    * Using the Class Names embeddings(text embedding) to form a class hierarchy.
    * Get a pretty higher result than existing methods.

- [ACMMM 2019] TGG: Transferable Graph Generation for Zero-shot and Few-shot Learning
    * using class-level knowledge graph to enhance instance-level graph
    * pretty interesting
    * did not compare on the mini-imagenet

- [CVPR 2020] Boosting Few-Shot Learning With Adaptive Margin Loss
    * using word embeddings to compute the relationship between classes and
        this relationship is used as guidance to learn the adaptive margin
        loss, sound reasonable.
    * 67.10 ± 0.52% and 79.54 ± 0.60% on mini-ImageNet

- [ECCV 2020] ([paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550715.pdf)) Large-Scale Few-Shot Learning via Multi-Modal Knowledge Discovery
    * Similar to CVPR 2019 Large-Scale Few-shot Learning Knowledge Transfer with Class Hierarchy

#### Self-training

- [NIPS 2019] ([paper](https://arxiv.org/pdf/1906.00562.pdf)) Learning to Self-Train for Semi-Supervised Few-Shot Classification
  - Label the query set for the first run, then retrain the model with the pseudo label for the second run. (Simple but effective)

- [ICLR 2021 oral] ([paper](https://arxiv.org/pdf/2010.07734.pdf)) SELF-TRAINING FOR FEW-SHOT TRANSFER ACROSS EXTREME TASK DIFFERENCES
  - Focusing on the task where the difference between base and novel is as extreme
  - They assume few-shot learning tech fail to bridge extreme difference and self-supervised tech fail when they ignore inductive biases from ImageNet
  - They pretrained on source domain, then use pre-trained backbone to grouping
      unlabeled samples then re-training on it.

### Results in Datasets
Basically, we use [Omniglot](https://github.com/brendenlake/omniglot), [mini-Imagenet](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning),
 [tiredImagenet](https://arxiv.org/abs/1803.00676), [CUB 2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and full [Imagenet](http://image-net.org) for the datasets. We list the latest methods' performs in mini-Imagenet.
Welcome contributes to expand the tables of results. 

基本上在小样本图像分类领域，主流的数据集为 Omniglot，mini-Imagenet，tired-Imagenet，CUB 和完整的 ImageNet。在这里我们总结了当前已有的方法在 mini-ImageNet 上的表现。
非常欢迎大家来补充呀。(鉴于精力有限，这部分的内容不再维护了，如果有小伙伴也愿意一起维护的话，可以联系我呀)

#### [mini-Imagenet](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning)
Basically, most methods achieve 70+% for 1-shot and 82+% for mini-ImageNet.

| Years | Methods              | Backbone | 5-way 1-shot    | 5-way 5-shot    |
|-------|----------------------|----------|-----------------|-----------------|
| 2016  | Matching Network     | Conv4    | 43.56 +- 0.84%  | 55.31% +- 0.73% |
| 2017  | MAML                 | Conv4    | 48.7% +- 1.84%  | 63.15% +- 0.91% |
| 2017  | Prototypical Network | Conv4    | 49.42% +- 0.78% | 68.20% +- 0.66% |
| 2017  | OPTIMIZATION AS A MODEL FOR FEW-SHOT LEARNING    |  Conv4   | 43.44+-0.77% | 60.60+-0.71% |
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
| 2019  | Centroid Networks for Few-Shot Clustering and Unsupervised Few-Shot Classification    |  Conv4   |  | 62.6+-0.5% |
| 2019  | Infinite Mixture Prototypes for Few-Shot Learning   |  Conv4   | 49.6+-0.8% | 68.1+-0.8% |
| 2020  | META-LEARNING WITH WARPED GRADIENT DESCENT |  Conv4   | 52.3 ± 0.8% | 68.4 ± 0.6% |
| 2021  | IEPT |  ResNet-12   | 67.05 ± 0.44% | 82.90 ± 0.30% |
