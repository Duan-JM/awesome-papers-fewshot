# Introdution
This part save the papers collected from arXiv, paper will be deleted if I
judge it is not good or not accpeted to the top conference over one year.

# Remain checkout
- [ ] [ICML 2020] On the Global Optimality of Model-Agnostic Meta-Learning
- [ ] [ICML 2020] Meta-learning for mixed linear regression
- [ ] [ICML 2020] A Sample Complexity Separation between Non-Convex and Convex Meta-Learning
- [ ] [ICML 2020] MetaFun: Meta-Learning with Iterative Functional Updates
- [ ] [ICML 2020] Learning Attentive Meta-Transfer
- [ ] [ICML 2020] Meta-learning with Stochastic Linear Bandits
- [ ] [ICML 2020] Meta-Learning with Shared Amortized Variational Inference
- [ ] [ICML 2020] TaskNorm: Rethinking Batch Normalization for Meta-Learning
- [ ] [ICML 2020] Meta Variance Transfer: Learning to Augment from the Others
- [ ] [ICML 2020] XtarNet: Learning to Extract Task-Adaptive Representation for Incremental Few-Shot Learning
- [ ] [ICML 2020] Few-shot Relation Extraction via Bayesian Meta-learning on Task Graph
- [ ] [ICML 2020] Unraveling Meta-Learning: Understanding Feature Representations for Few-Shot Tasks
- [ ] [ICML 2020] Laplacian Regularized Few-Shot Learning

# Remain Read Papers
- [arXiv 2020] Improving out-of-distribution generalization via multi-task self-supervised pretraining
    * finished 
    * no compare to mini-imagenet, so hard to compare
    * exist code

# Other Conference
- [x] [WACV 2020] Charting the Right Manifold: Manifold Mixup for Few-shot Learning
- [x] [ESANN 2020] Zero-shot and few-shot time series forecasting with ordinal regression recurrent neural networks
- [x] [CVPR 2020 Workshop] MA 3 : Model Agnostic Adversarial Augmentation for Few Shot learning
- [ ] [Neuro Computing 2020] Revisiting Metric Learning for Few-Shot Image Classification
- [ ] [IJCNN 2020] RelationNet2: Deep Comparison Columns for Few-Shot Learning
- [ ] [CVPR 2020 Workshop] Meta-Learning for Few-Shot Land Cover Classification
- [x] [OCEANS 2020] A Comparison of Few-Shot Learning Methods for Underwater Optical and Sonar Image Classification
    * K-means enhance the prototypes, similar to my previous papers.
- [ ] [InterSpeech 2020] AdaDurIAN: Few-shot Adaptation for Neural Text-to-Speech with DurIAN

# Arxiv
## Summary
- [arXiv 2020] A Concise Review of Recent Few-shot Meta-learning Methods
    * Change the Methods into four methods. (Basically exclude metric-based such as ProtoNet)
        * Learning an Initialization
        * Generation of Parameters
        * Learning an Optimizer (doubt for this, maybe sort to the second)
        * Memory-based Methods

- [x] [arXiv 2019] A New Benchmark for Evaluation of Cross-Domain Few-Shot Learning

## Image Classification
- [x] [arXiv 2019] Dont Even Look Once: Synthesizing Features for Zero-Shot Detection
- [x] [arXiv 2019] Learning Generalizable Representations via Diverse Supervision

- [x] [arXiv 2019] Auxiliary Learning for Deep Multi-task Learning 
    * 解决multitask 参数共享问题的

- [x] [arXiv 2019] All you need is a good representation: A multi-level and classifier-centric representation for few-shot learning (一般)
- [x] [arXiv 2019] A Multi-Task Gradient Descent Method for Multi-Label Learning
- [x] [arXiv 2019] Lifelong Spectral Clustering 
    * 连续学习、聚类后期对信息的存储

- [x] [arXiv 2019] CNN-based Dual-Chain Models for Knowledge Graph Learning
- [x] [arXiv 2019] MetAdapt: Meta-Learned Task-Adaptive Architecture for Few-Shot Classification
    * 使用模型搜索搜出来的结构，号称 SOTA 在 mini-imagenet （存疑）

    * 这个是在feature上动文章的，关键词是self-supervised 和 regularization technique。This work investigates the role of learning relevant feature manifold for few-shot tasks using self-supervision and regularization techniques.

- [x] [arXiv 2019] MetaFun: Meta-Learning with Iterative Functional Updates
    * 用了无限的特征长度，还有一个什么东西，效果很好83%

- [x] [arXiv 2019] Semantic Regularization: Improve Few-shot Image Classification by Reducing Meta Shift
- [x] [1909.11446] Decoder Choice Network for Meta-Learning
    * leo分支的、目标是减参数，效果一般，可以看看怎么减少参数的
- [x] [arXiv 2018] Few-Shot Self Reminder to Overcome Catastrophic Forgetting
    * Ultilize the loss between tasks
    * Ultilize the generate feature betwwen tasks
- [x] [arXiv 2019] Diversity Transfer Network for Few-Shot Learning
    * ResNet12 77.91%
- [x] [arXiv 2019] FLAT: Few-Shot Learning via Autoencoding Transformation Regularizers
    * ResNet18 77%
- [x] [arXiv 2019] ([paper](https://arxiv.org/abs/1911.06045) [code](https://github.com/phecy/SSL-FEW-SHOT)) SELF-SUPERVISED LEARNING FOR FEW-SHOT IMAGE CLASSIFICATION
    * 90% accuracy

- [x] [arXiv 2020] ([paper](https://arxiv.org/pdf/2001.09849.pdf) [code](https://github.com/yhu01/transfer-sgc)) Exploiting Unsupervised Inputs for Accurate Few-Shot Classification
    * 85% Graph
    * wait author to refine the paper

- [x] [arXiv 2019] ([paper](https://arxiv.org/abs/1906.02944) ) Learning Adaptive Classifiers Synthesis for Generalized Few-Shot Learning
- [x] [arXiv 2019] ([paper](https://arxiv.org/pdf/2001.08366.pdf)) Continual Local Replacement for Few-shot Image Recognition
    * using generative model to generate image to replace support set locally(pretty fun)
    * 66 1shot 81 5-shot

- [x] [arXiv 2020] ([paper](https://arxiv.org/pdf/1812.04955.pdf)) Prior-Knowledge and Attention based Meta-Learning for Few-Shot Learning
    * add Very Very simple attention(almost like SENet's attention model)
    * add addition model to assist
    * 1 per improve on PN

- [x] [arXiv 2020] ([paper](https://arxiv.org/pdf/2002.07522.pdf)) Few-Shot Few-Shot Learning and the role of Spatial Attention
    * reported 80% acc
    * Interesting
- [x] [arXiv 2020] An Ensemble of Epoch-wise Empirical Bayes for Few-shot Learning
    * addon on the SIB(semi) 81% on 5shot in miniImagnet
- [x] [arXiv 2020] StarNet: towards weakly supervised few-shot detection and explainable few-shot classification
    * explanable few-shot classification
    * 79% on mini-imagenet
- [x] [arXiv 2020] TAFSSL: Task-Adaptive Feature Sub-Space Learning for few-shot classification
    * 77% for 1shot and 84.99 for 5-shot on mini-Imagenet
- [x] [arXiv 2020] Associative Alignment for Few-shot Image Classification
    * assit with base instances
- [x] [arXiv 2020] Unraveling Meta-Learning: Understanding Feature Representations for Few-Shot Tasks
- [x] [arXiv 2020] Selecting Relevant Features from a Universal Representation for Few-shot Classification
    * mixup the final feature with multi model output or multi depth layer in the models output
- [x] [arXiv 2020] Few-Shot Learning with Geometric Constraints
    * main contribution is on remain accurarcy for both novel and base
    * outperform in both situations on miniImagenet
- [x] [arXiv 2020] ([code](http://github.com/WangYueFt/rfs/)) Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?
    * finetune is all you need
    * 62.02% or 64.82% 1-shot and 79.64% or 82.14% 5shot
- [x] [arXiv 2020] A New Meta-Baseline for Few-Shot Learning
- [ ] [arXiv 2020] [exist code] Negative Margin Matters: Understanding Margin in Few-shot Classification
    * 62% for 1shot
- [x] [arXiv 2020] AdarGCN: Adaptive Aggregation GCN for Few-Shot Learning
    * a new circumstance with noise input
- [x] [arXiv 2020] Prototype Rectification for Few-Shot Learning
    * Using query set to enhance prototype, good results in 1-shot 70%, however, potential
        model leaky problem, wait for opensource
- [x] [arXiv 2020] [exist code] Embedding Propagation: Smoother Manifold for Few-Shot Classification
    * ElementAI
    * 70 1-shot 81 5-shot
- [x] [arXiv 2020] TransMatch: A Transfer-Learning Scheme for Semi-Supervised Few-Shot Learning
    * Imprinted weights to finetune 
    * SSL settings
    * 63% 1-shot 82% 5shot
- [x] [arXiv 2020] [exist code] Transductive Few-shot Learning with Meta-Learned Confidence
    * 78 1-shot 86 5-shot
- [x] [arXiv 2020] Self-Augmentation: Generalizing Deep Networks to Unseen Classes for Few-Shot Learning
    * 65 1-shot 82 5-shot 
    * deep metric learning & cutmix
- [x] [arXiv 2020] Embedding Propagation: Smoother Manifold for Few-Shot Classification
    * rotation as self-supervised, not impressive
    * 83% for 5 shot under SSL settings
- [ ] [arXiv 2020] Unsupervised Few-shot Learning via Distribution Shift-based Augmentation
- [ ] [arXiv 2020] Meta-Meta-Classification for One-Shot Learning
- [ ] [arXiv 2020] Divergent Search for Few-Shot Image Classification
- [ ] [arXiv 2020] Physarum Powered Differentiable Linear Programming Layers and Applications
    * An plug and play layer, FC-100 improve Cifar-100 FS for 1% on MetaOptSVM
- [x] [arXiv 2020] Generalized Reinforcement Meta Learning for Few-Shot Optimization
    * This paper's motivation like "Empirical Bayes Transductive Meta-Learning with Synthetic Gradients in ICLR 2020, both of them use a mechanism to estimate or synthesis the gradients, so if you interesting in this paper, you'd better have look it that one.
    * 71% with Resnet on mini-Imagenet (not so impressive)
- [ ] [arXiv 2020] Bayesian Online Meta-Learning with Laplace Approximation
    * continue learning
- [x] [arXiv 2020] ONE OF THESE (FEW) THINGS IS NOT LIKE THE OTHERS
    * image classification task, however need to classify the outliner, like cross-domain settings
    * they define a "junk" classes specific, may not significant
- [x] [arXiv 2020] Compositional Few-Shot Recognition with Primitive Discovery and Enhancing
    * 63.21 ± 0.78% for 1-shot on mini-ImageNet
    * insterest, they improve the results by decompose the image into several compostions, sound reasonable
- [x] [arXiv 2020] Looking back to lower-level information in few-shot learning
    * arguing that output of each layer in the backbone's could use to learn
    * build TPN on each layer of the graph
    * Improve TPN for 1~2% for mini-Imagenet and tiered-Imagenet compare to TPN

- [x] [arXiv 2020] TOAN: Target-Oriented Alignment Network for Fine-Grained Image Categorization with Few Labeled Samples
    * very similar to one of work in CVPR 2019 -> Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning
    * Not compare in mini-ImageNet
    * same to "Low-Rank Pairwise Alignment Bilinear Network For Few-Shot Fine-Grained Image Classification"

- [ ] [arXiv 2020] Few-shot Learning for Domain-specific Fine-grained Image Classification
- [ ] [arXiv 2020] Distributionally Robust $k$-Nearest Neighbors for Few-Shot Learning
    * Seem mathmatically methods, will verify it later

**Generation**
- [x] [arXiv 2020] MatchingGAN: Matching-based Few-shot Image Generation
- [x] [arXiv 2020] Few-shot Compositional Font Generation with Dual Memory

**Object Detection & Tracking**
- [x] [arXiv 2019] One-Shot Object Detection with Co-Attention and Co-Excitation
    * senet的迁移
- [x] [arXiv 2020] Context-Transformer: Tackling Object Confusion for Few-Shot Detection
- [x] [arXiv 2020] Meta R-CNN : Towards General Solver for Instance-level Few-shot Learning
- [x] [arXiv 2020] Any-Shot Object Detection
- [ ] [arXiv 2020] Incremental Few-Shot Object Detection
- [x] [arXiv 2020] MOTS: Multiple Object Tracking for General Categories Based On Few-Shot Method
    * nearly same to prototype networks

**Segmentation**
- [ ] [arXiv 2020] CRNet: Cross-Reference Networks for Few-Shot Segmentation
- [ ] [arXiv 2020] On the Texture Bias for Few-Shot CNN Segmentation
- [ ] [arXiv 2020] [exist code] Learning to Segment the Tail
- [ ] [arXiv 2020] Semi-supervised few-shot learning for medical image segmentation
- [ ] [arXiv 2020] Objectness-Aware One-Shot Semantic Segmentation
- [ ] [arXiv 2020] Self-Supervised Tuning for Few-Shot Segmentation
- [ ] [arXiv 2020] Prototype Refinement Network for Few-Shot Segmentation

**NLP**
- [ ] [arXiv 2020] Few-shot Natural Language Generation for Task-Oriented Dialog
- [ ] [arXiv 2020] SOLOIST: Few-shot Task-Oriented Dialog with A Single Pre-trained Auto-regressive Model
- [x] [arXiv 2020] Prototypical Q Networks for Automatic Conversational Diagnosis and Few-Shot New Disease Adaption
    * Using a dialog to simulate patient and doctors' conversition to finally
        give a diagnosis

- [x] [arXiv 2020] BOFFIN TTS: FEW-SHOT SPEAKER ADAPTATION BY BAYESIAN OPTIMIZATION
    * Text to Speech
- [ ] [arXiv 2020] MICK: A Meta-Learning Framework for Few-shot Relation Classification with Little Training Data
    * Relation Classification
- [ ] [arXiv 2020] Logic2Text: High-Fidelity Natural Language Generation from Logical Forms
- [ ] [arXiv 2020] Learning to Few-Shot Learn Across Diverse Natural Language Classification Tasks
    * LEO 风格的文本分类
    * Structure like LEO, named LEOPARD
- [ ] [arXiv 2020] Meta Fine-Tuning Neural Language Models for Multi-Domain Text Mining
- [ ] [arXiv 2020] Cross-lingual Zero- and Few-shot Hate Speech Detection Utilising Frozen Transformer Language Models and AXEL
- [ ] [arXiv 2020] Few-Shot Natural Language Generation by Rewriting Templates
- [ ] [arXiv 2020] Pay Attention to What You Read: Non-recurrent Handwritten Text-Line Recognition
- [ ] [arXiv 2020] Few-shot Text Classification with Distributional Signatures
- [ ] [arXiv 2020] Knowledge Guided Metric Learning for Few-Shot Text Classification
- [ ] [arXiv 2020] CG-BERT: Conditional Text Generation with BERT for Generalized Few-shot Intent Detection

**Cross-Domain**
- [ ] [arXiv 2020] Towards Fair Cross-Domain Adaptation via Generative Learning
- [ ] [arXiv 2020] Cross-Domain Few-Shot Learning with Meta Fine-Tuning
- [ ] [arXiv 2020] Feature Transformation Ensemble Model with Batch Spectral Regularization for Cross-Domain Few-Shot Classification
- [ ] [arXiv 2020] Cross-Domain Few-Shot Learning with Meta Fine-Tuning
- [ ] [arXiv 2020] Few-Shot Open-Set Recognition using Meta-Learning
- [ ] [arXiv 2020] Large Margin Mechanism and Pseudo Query Set on Cross-Domain Few-Shot Learning
- [ ] [arXiv 2020] M3P: Learning Universal Representations via Multitask Multilingual Multimodal Pre-training
- [ ] [arXiv 2020] Few-Shot Open-Set Recognition using Meta-Learning

**Uncertainty**
- [ ] [arXiv 2020] Calibrated neighborhood aware confidence measure for deep metric learning
    * The approach approximates the distribution of data points for each class using a Gaussian kernel smoothing function.
    * They sperate the uncertainty measure methods into three branches Calibration on the held-out validation data, Bayesian approximation / Support set based uncertainnty estimation
    * The gt used for uncertainty is measured by eucildean distances and I
        doublt the uncertainty measured by eucildean in high dimensional spaces
        is accurate.

**Application**
- [x] [arXiv 2019] Learning Predicates as Functions to Enable Few-shot Scene Graph Prediction
- [x] [arXiv 2019] Real-Time Object Tracking via Meta-Learning: Efficient Model Adaptation and One-Shot Channel Pruning （目标跟踪）
- [x] [arXiv 2019] Defensive Few-shot Adversarial Learning
- [x] [arXiv 2019] Few-shot Learning with Contextual Cueing for Object Recognition in Complex Scenes
- [x] [arXiv 2019] Meta-Learning with Dynamic-Memory-Based Prototypical Network
- [x] [arXiv 2020] DAWSON: A Domain Adaptive Few Shot Generation Framework
-   * generate music, a project under cs236 in stanford university

- [x] [arXiv 2019] ADVERSARIALLY ROBUST FEW-SHOT LEARNING: A META-LEARNING APPROACH
    * A approach is robust to adversarially attack

- [x] [arXiv 2020] Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning
- [ ] [arXiv 2020] Meta-Learning Initializations for Low-Resource Drug Discovery
- [ ] [arXiv 2020] An Open-set Recognition and Few-Shot Learning Dataset for Audio Event Classification in Domestic Environments
- [ ] [arXiv 2020] Additive Angular Margin for Few Shot Learning to Classify Clinical Endoscopy Images
- [ ] [arXiv 2020] Domain-Adaptive Few-Shot Learning
- [ ] [arXiv 2020] Efficient Intent Detection with Dual Sentence Encoders
- [ ] [arXiv 2020] Zero-Shot Cross-Lingual Transfer with Meta Learning
- [ ] [arXiv 2020] From Zero to Hero: On the Limitations of Zero-Shot Cross-Lingual Transfer with Multilingual Transformers
- [ ] [arXiv 2020] An Open-set Recognition and Few-Shot Learning Dataset for Audio Event Classification in Domestic Environments
- [ ] [arXiv 2020 wip] PAC-BAYESIAN META-LEARNING WITH IMPLICIT PRIOR
    * 63 for 1shot, 78 for 5-shot
    * LEO branch
- [ ] [arXiv 2020] Revisiting Few-shot Activity Detection with Class Similarity Control
- [ ] [arXiv 2020] Meta-Learning for Few-Shot NMT Adaptation
- [ ] [arXiv 2020] SSHFD: Single Shot Human Fall Detection with Occluded Joints Resilience
- [ ] [arXiv 2020] Gradient-based Data Augmentation for Semi-Supervised Learning
- [ ] [arXiv 2020] Few-Shot Single-View 3-D Object Reconstruction with Compositional Priors
- [ ] [arXiv 2020] Alleviating the Incompatibility between Cross Entropy Loss and Episode Training for Few-shot Skin Disease Classification
- [ ] [arXIv 2020] TAEN: Temporal Aware Embedding Network for Few-Shot Action Recognition
- [ ] [arXiv 2020] Signal Level Deep Metric Learning for Multimodal One-Shot Action Recognition
- [ ] [arXiv 2020] ST2: Small-data Text Style Transfer via Multi-task Meta-Learning
- [ ] [arXiv 2020] Learning to Classify Intents and Slot Labels Given a Handful of Examples
- [ ] [arXiv 2020] PuzzLing Machines: A Challenge on Learning From Small Data
- [ ] [arXiv 2020] Learning to Learn to Disambiguate: Meta-Learning for Few-Shot Word Sense Disambiguation
- [ ] [arXiv 2020] Few-Shot Learning for Abstractive Multi-Document Opinion Summarization
- [ ] [arXiv 2020] Interactive Video Stylization Using Few-Shot Patch-Based Training
- [ ] [arXiv 2020] MAD-X: An Adapter-based Framework for Multi-task Cross-lingual Transfer
- [ ] [arXiv 2020] Self-Training with Improved Regularization for Few-Shot Chest X-Ray Classification
- [ ] [arXiv 2020] 3FabRec: Fast Few-shot Face alignment by Reconstruction
    * facial landmark detection
- [ ] [arXiv 2020] Combining Deep Learning with Geometric Features for Image based Localization in the Gastrointestinal Tract
- [ ] [arXiv 2020] Meta-Learning of Neural Architectures for Few-Shot Learning
- [ ] [arXiv 2020] SSM-Net for Plants Disease Identification in LowData Regime
    * disease in agricultural
- [ ] [arXiv 2020] Interpretable Time-series Classification on Few-shot Samples
- [ ] [arXiv 2020] Learning to Extrapolate Knowledge: Transductive Few-shot Out-of-Graph Link Prediction
    * out-of-graph link prediction task
