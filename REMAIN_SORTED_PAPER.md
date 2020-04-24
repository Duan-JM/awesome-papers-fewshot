# No Papers
- [ ] [AAAI 2020] SGAP-Net: Semantic-Guided Attentive Prototypes Network for Few-Shot Human-Object Interaction Recognition

# Remain Read Papers
- [arXiv 2020] Improving out-of-distribution generalization via multi-task self-supervised pretraining
    * finished 
    * no compare to mini-imagenet, so hard to compare
    * exist code

# Not so Top Conference
- [x] [WACV 2020] Charting the Right Manifold: Manifold Mixup for Few-shot Learning
- [x] [ESANN 2020] Zero-shot and few-shot time series forecasting with ordinal regression recurrent neural networks
- [x] [CVPR 2020 Workshop] MA 3 : Model Agnostic Adversarial Augmentation for Few Shot learning
- [ ] [Neuro Computing 2020] Revisiting Metric Learning for Few-Shot Image Classification

# Arxiv
- [x] [arXiv 2019] Dont Even Look Once: Synthesizing Features for Zero-Shot Detection
- [x] [arXiv 2019] Learning Generalizable Representations via Diverse Supervision
- [x] [arXiv 2019] One-Shot Object Detection with Co-Attention and Co-Excitation
    * senet的迁移

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

- [x] [arXiv 2019] Class Regularization-Improve Few-shot Image Classification by Reducing Meta Shift
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

- [x] [arXiv 2020] ([paper](https://arxiv.org/pdf/2001.09849.pdf) [code](https://github.com/yhu01/transfer-sgc)) Exploiting Unsupervised Inputs for Accurate Few-Shot Classification
    * 85% Graph
    * wait author to refine the paper

- [x] [arXiv 2019] ([paper](https://arxiv.org/abs/1906.02944) ) Learning Adaptive Classifiers Synthesis for Generalized Few-Shot Learning
- [x] [arXiv 2019] ([paper](https://arxiv.org/pdf/2001.08366.pdf)) Continual Local Replacement for Few-shot Image Recognition
    * using generative model to generate iamge to replace support set locally(pretty fun)
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
- [x] [arXiv 2020] Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?
    * finetune is all you need
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
    * deep metric learninng & cutmix
- [x] [arXiv 2020] Embedding Propagation: Smoother Manifold for Few-Shot Classification
    * rotation as self-supervised, not impressive
    * 83% for 5 shot under SSL settings
- [ ] [arXiv 2020] Unsupervised Few-shot Learning via Distribution Shift-based Augmentation
- [ ] [arXiv 2020] Meta-Meta-Classification for One-Shot Learning
- [ ] [arXiv 2020] Divergent Search for Few-Shot Image Classification

**Generation**
- [x] [arXiv 2020] MatchingGAN: Matching-based Few-shot Image Generation

**Object Detection**
- [x] [arXiv 2020] Context-Transformer: Tackling Object Confusion for Few-Shot Detection
- [x] [arXiv 2020] Meta R-CNN : Towards General Solver for Instance-level Few-shot Learning
- [x] [arXiv 2020] Any-Shot Object Detection
- [x] [arXiv 2020] Frustratingly Simple Few-Shot Object Detection
    * say fintune is outperform meta-learning methods
- [ ] [arXiv 2020] Incremental Few-Shot Object Detection

**Segmentation**
- [ ] [arXiv 2020] CRNet: Cross-Reference Networks for Few-Shot Segmentation
- [ ] [arXiv 2020] On the Texture Bias for Few-Shot CNN Segmentation
- [ ] [arXiv 2020] [exist code] Learning to Segment the Tail
- [ ] [arXiv 2020] Semi-supervised few-shot learning for medical image segmentation
- [ ] [arXiv 2020] Objectness-Aware One-Shot Semantic Segmentation
- [ ] [arXiv 2020] Self-Supervised Tuning for Few-Shot Segmentation

**Application**
- [x] [arXiv 2019] Learning Predicates as Functions to Enable Few-shot Scene Graph Prediction
- [x] [arXiv 2019] Real-Time Object Tracking via Meta-Learning: Efficient Model Adaptation and One-Shot Channel Pruning （目标跟踪）
- [x] [arXiv 2019] Defensive Few-shot Adversarial Learning
- [x] [arXiv 2019] A New Benchmark for Evaluation of Cross-Domain Few-Shot Learning
- [x] [arXiv 2019] Few-shot Learning with Contextual Cueing for Object Recognition in Complex Scenes
- [x] [arXiv 2019] Meta-Learning with Dynamic-Memory-Based Prototypical Network
- [x] [arXiv 2020] DAWSON: A Domain Adaptive Few Shot Generation Framework
-   * generate music, a project under cs236 in stanford university
- [x] [arXiv 2020] BOFFIN TTS: FEW-SHOT SPEAKER ADAPTATION BY BAYESIAN OPTIMIZATION
    * Text to Speech

- [x] [arXiv 2020] Few-Shot Learning as Domain Adaptation: Algorithm and Analysis
- [x] [arXiv 2020] ([paper](https://arxiv.org/pdf/2001.08735.pdf))CROSS-DOMAIN FEW-SHOT CLASSIFICATION VIA LEARNED FEATURE-WISE TRANSFORMATION
- [x] [arXiv 2019] ADVERSARIALLY ROBUST FEW-SHOT LEARNING: A META-LEARNING APPROACH
    * A approach is robust to adversarially attack

- [x] [arXiv 2019] Shaping Visual Representations with Language for Few-shot Classification
    * jointly predicting natural language task descriptions
- [x] [arXiv 2020] Few-Shot Learning as Domain Adaptation: Algorithm and Analysis
- [x] [arXiv 2020] Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning
- [ ] [arXiv 2020] Meta-Learning Initializations for Low-Resource Drug Discovery
- [ ] [arXiv 2020] An Open-set Recognition and Few-Shot Learning Dataset for Audio Event Classification in Domestic Environments
- [ ] [arXiv 2020] Additive Angular Margin for Few Shot Learning to Classify Clinical Endoscopy Images
- [ ] [arXiv 2020] Domain-Adaptive Few-Shot Learning
- [ ] [arXiv 2020] Efficient Intent Detection with Dual Sentence Encoders
- [ ] [arXiv 2020] Zero-Shot Cross-Lingual Transfer with Meta Learning
- [ ] [arXiv 2020] Towards Fair Cross-Domain Adaptation via Generative Learning
- [ ] [arXiv 2020] Learning to Few-Shot Learn Across Diverse Natural Language Classification Tasks
    * LEO 风格的文本分类
    * Structure like LEO, named LEOPARD
- [ ] [arXiv 2020] An Open-set Recognition and Few-Shot Learning Dataset for Audio Event Classification in Domestic Environments
- [ ] [arXiv 2020] Few-shot Natural Language Generation for Task-Oriented Dialog
- [ ] [arXiv 2020 wip] PAC-BAYESIAN META-LEARNING WITH IMPLICIT PRIOR
    * 63 for 1shot, 78 for 5-shot
    * LEO branch
- [ ] [arXiv 2020] Revisiting Few-shot Activity Detection with Class Similarity Control
- [ ] [arXiv 2020] Meta Fine-Tuning Neural Language Models for Multi-Domain Text Mining
- [ ] [arXiv 2020] Meta-Learning for Few-Shot NMT Adaptation
- [ ] [arXiv 2020] Knowledge Guided Metric Learning for Few-Shot Text Classification
- [ ] [arXiv 2020] CG-BERT: Conditional Text Generation with BERT for Generalized Few-shot Intent Detection
- [ ] [arXiv 2020] SSHFD: Single Shot Human Fall Detection with Occluded Joints Resilience
- [ ] [arXiv 2020] Gradient-based Data Augmentation for Semi-Supervised Learning
- [ ] [arXiv 2020] Few-Shot Single-View 3-D Object Reconstruction with Compositional Priors
- [ ] [arXiv 2020] Alleviating the Incompatibility between Cross Entropy Loss and Episode Training for Few-shot Skin Disease Classification
- [ ] [arXIv 2020] TAEN: Temporal Aware Embedding Network for Few-Shot Action Recognition
