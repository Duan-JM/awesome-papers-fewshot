<!-- vim-markdown-toc GitLab -->

* [Domain Adaptation](#domain-adaptation)
* [Reinforcement Learning](#reinforcement-learning)
* [Visual Tracking](#visual-tracking)
* [Theoritical](#theoritical)
* [Recommendation System](#recommendation-system)
* [Graph Classification](#graph-classification)
* [Others](#others)

<!-- vim-markdown-toc -->
### Domain Adaptation
- [NIPS 2017] Few-Shot Adversarial Domain Adaptation
- [ICCV 2019] Bidirectional One-Shot Unsupervised Domain Mapping
- [ICLR 2020 spotlight] ([paper](https://arxiv.org/pdf/2001.08735.pdf) ) Cross-domain Few-shot Classification via Learned Feature-wise Transformation 
    * Plug in addon parameters to adjust between unseen and seen encoders.
    * Training in pair-wise sample strategy between seen domain and unseen domain.
- [ICLR 2020] Meta-learning curiosity algorithms
- [ICLR 2020] Meta Reinforcement Learning with Autonomous Inference of Subtask Dependencies
- [ICML 2020] ([paper](https://arxiv.org/abs/2002.03497)) Few-shot Domain Adaptation by Causal Mechanism Transfer
- [NIPS 2020] CrossTransformers: spatially-aware few-shot transfer
    * Using transformer as encoder and query images performed as query keys
- [ICLR 2021] ([paper](https://openreview.net/pdf?id=04cII6MumYV)) A UNIVERSAL REPRESENTATION TRANSFORMER LAYER FOR FEW-SHOT IMAGE CLASSIFICATION
    * multi-domain few-shot image classification
- [ICLR 2021] ([paper](https://openreview.net/pdf?id=qkLMTphG5-h)) REPURPOSING PRETRAINED MODELS FOR ROBUST OUT-OF-DOMAIN FEW-SHOT LEARNING
- [ICLR 2021] ([paper](https://arxiv.org/abs/2103.12051)) SSD: A Unified Framework for Self-Supervised Outlier Detection

### Reinforcement Learning
- [ICML 2019] Few-Shot Intent Inference via Meta-Inverse Reinforcement Learning
- [ICLR 2020] Meta Reinforcement Learning with Autonomous Inference of Subtask Dependencies
- [ICLR 2020] VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning
- [NIPS 2020] ([paper](https://arxiv.org/abs/2010.14484)) One Solution is Not All You Need: Few-Shot Extrapolation via Structured MaxEnt RL

### Visual Tracking
- [ICCV 2019] Deep Meta Learning for Real-Time Target-Aware Visual Tracking
- [CVPR 2020] ([paper](https://arxiv.org/abs/2004.00830)) Tracking by Instance Detection: A Meta-Learning Approach
    * MAML-Tracker

### Theoritical
- [ICLR 2020 Bengio] A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms
    * Argue that how to fast adapt to new distributions by meta-learn causal structures
    * Also have follow paper on arxiv [here](https://www.semanticscholar.org/paper/An-Analysis-of-the-Adaptation-Speed-of-Causal-Priol-Harikandeh/982816b03c2f09f7eb63f40dfbedd03caa3e5570?utm_source=alert_email&utm_content=AuthorPaper&utm_campaign=AlertEmails_DAILY&utm_term=AuthorPaper&email_index=0-0-0&utm_medium=244646)

### Recommendation System
- [SKIM 2020] Learning to Profile: User Meta-Profile Network for Few-Shot Learning
    * Interesting Time sequence encoding methods
    * Using meta-learning methods to learn user-profile representations
    * Can be used to solve data scarcity or class imbalance problem

### Graph Classification
- [CIKM 2020] Adaptive-Step Graph Meta-Learner for Few-Shot Graph Classification
- [CIKM 2020] Graph Prototypical Networks for Few-shot Learning on Attributed Networks
    * Graph node classification
- [CIKM 2020] Graph Few-shot Learning with Attribute Matching
- [NIPS 2020] ([paper](https://arxiv.org/pdf/2007.02914.pdf)) Node Classification on Graphs with Few-Shot Novel Labels via Meta Transformed Network Embedding

### Others
- [IJCAI 2019] Incremental Few-Shot Learning for Pedestrian Attribute Recognition
- [AAAI 2018] AffinityNet- Semi-supervised Few-shot Learning for Disease Type Prediction Use few-shot method to enhance urinal disease type prediction

- [NIPS 2018] Neural Voice Cloning with a Few Samples
- [ICCV 2019] ([paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_ACMM_Aligned_Cross-Modal_Memory_for_Few-Shot_Image_and_Sentence_Matching_ICCV_2019_paper.pdf)) ACMM: Aligned Cross-Modal Memory for Few-Shot Image and Sentence Matching
    * Image and Sentence Matching
- [ICCV 2019] (RECOMMANDED!) Task-Driven Modular Networks for Zero-Shot Compositional Learning
    * An interesting usage of a bunch of MLPs.
- [CVPR 2019] ([paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Rostami_SAR_Image_Classification_Using_Few-Shot_Cross-Domain_Transfer_Learning_CVPRW_2019_paper.pdf) [code](https://github.com/MSiam/AdaptiveMaskedProxies.)) SAR Image Classification Using Few-shot Cross-domain Transfer Learning
- [ICLR 2020] FEW-SHOT TEXT CLASSIFICATION WITH DISTRIBUTIONAL SIGNATURES
- [ICLR 2020] METAPIX: FEW-SHOT VIDEO RETARGETING
- [ICLR 2020] ONE-SHOT PRUNING OF RECURRENT NEURAL NETWORKS BY JACOBIAN SPECTRUM EVALUATION
    * Pruning
- [ICLR 2020] TOWARDS FAST ADAPTATION OF NEURAL ARCHITECTURES WITH META LEARNING
    * NAS
- [ICLR 2020] META-DATASET: A DATASET OF DATASETS FOR LEARNING TO LEARN FROM FEW EXAMPLES
    * New datasets
- [ECCV 2018] Few-Shot Human Motion Prediction via Meta-Learning
- [ECCV 2018] Compound Memory Networks for Few-shot Video Classification
- [CVPR 2019] Coloring With Limited Data: Few-Shot Colorization via Memory Augmented Networks
- [CVPR 2019] Improving Few-Shot User-Specific Gaze Adaptation via Gaze Redirection Synthesis
- [CVPR 2019] Learning to Learn How to Learn: Self-Adaptive Visual Navigation Using Meta-Learning
- [ICCV 2019] Few-Shot Adaptive Gaze Estimation
- [ICCV 2019] One-Shot Neural Architecture Search via Self-Evaluated Template Network
- [AAAI 2020] Learning Meta Model for Zero- and Few-shot Face Anti-spoofing
- [AAAI 2020] Graph Few-shot Learning via Knowledge Transfer
- [AAAI 2020] Few Shot Network Compression via Cross Distillation (模型压缩)
- [AAAI 2020] Few-Shot Bayesian Imitation Learning with Logical Program Policies
- [CVPR 2020] Meta-Transfer Learning for Zero-Shot Super-Resolution
- [CVPR 2020] Learning from Web Data with Self-Organizing Memory Module
    * solve label noise and background noise in the images with memory module [CVPR 2020] Single-view view synthesis with multi plane images
- [AAAI 2020] ([paper](https://aaai.org/Papers/AAAI/2020GB/AAAI-JiZ.4799.pdf)) SGAP-Net: Semantic-Guided Attentive Prototypes Network for Few-Shot Human-Object Interaction Recognition
- [ICLR 2020] MetaPix: Few-Shot Video Retargeting
- [ICLR 2020] ([paper](https://openreview.net/forum?id=r1eowANFvr)) Towards Fast Adaptation of Neural Architectures with Meta Learning 
- [ICLR 2020] Query-efficient Meta Attack to Deep Neural Networks
- [SIGGRAPH Asia 2019] Artistic Glyph Image Synthesis via One-Stage Few-Shot Learning
- [IJCNN 2020] Interpretable Time-series Classification on Few-shot Samples
- [CVPR 2018] ([paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Temporal_Hallucinating_for_CVPR_2018_paper.pdf)) Temporal Hallucinating for Action Recognition with Few Still Images
    * Attempt to recall cues from relevant action videos.
    * Maybe good at one-shot, not worse than the baseline in 5-shot and 10-shot scenarios.
- [ECCV 2020] n-Reference Transfer Learning for Saliency Prediction
- [ICML 2020] Meta-learning with Stochastic Linear Bandits
    * Linear Bandits itself is a task
- [ECCV 2020 spotlight] Few-shot Action Recognition via Permutation-invariant Attention
- [ECCV 2020 spotlight] Few-Shot Scene-Adaptive Anomaly Detection
- [NIPS 2020] Self-Supervised Few-Shot Learning on Point Clouds
- [MICAI 2020] ([paper](https://arxiv.org/abs/2008.07724)) Domain Generalizer: A Few-shot Meta Learning Framework for Domain Generalization in Medical Imaging
- [ICML 2020] ([paper](https://arxiv.org/abs/2008.02952)) Informative Dropout for Robust Representation Learning: A Shape-bias Perspective
- [IEEE IROS 2020] ([paper](https://arxiv.org/abs/2008.00819)) Tell me what this is: Few-Shot Incremental Object Learning by a Robot
- [CIKM 2020] Few-shot Insider Threat Detection
- [ACMMM 2020] Depth Guided Adaptive Meta-Fusion Network for Few-shot Video Recognition
- [NIPS 2020] Few-shot Visual Reasoning with Meta-Analogical Contrastive Learning
    * Visual Reasoning
- [NIPS 2020] OOD-MAML: Meta-Learning for Few-Shot Out-of-Distribution Detection and Classification
- [NIPS 2020] Adversarially Robust Few-Shot Learning: A Meta-Learning Approach
    * Adversarially Robust FSL
- [CVPR 2020] Multi-Domain Learning for Accurate and Few-Shot Color Constancy
- [CVPR 2020] Few-Shot Video Classification via Temporal Alignment
- [CVPR 2020] Few-Shot Pill Recognition
- [CVPR 2020] Learning to Select Base Classes for Few-shot Classification
- [CVPR 2020] Few-Shot Open-Set Recognition using Meta-Learning
- [NIPS 2020] ([paper](https://arxiv.org/abs/2012.02721)) Event Guided Denoising for Multilingual Relation Learning
- [AAAI 2021] Progressive Network Grafting for Few-Shot Knowledge Distillation
- [NIPS 2020] ADVERSARIALLY ROBUST FEW-SHOT LEARNING: A META-LEARNING APPROACH
    * A approach is robust to adversarial attack
- [ICLR 2021] ([paper](https://openreview.net/pdf?id=bJxgv5C3sYc)) FEW-SHOT BAYESIAN OPTIMIZATION WITH DEEP KERNEL SURROGATES
    * Hyper Parameter Optimization
- [TPMAI 2021] ([paper](https://arxiv.org/pdf/1907.09382.pdf)) Domain-Specific Priors and Meta Learning for Few-Shot First-Person Action Recognition
- [WWW 2021] ([paper](https://arxiv.org/pdf/2102.07916.pdf)) Few-Shot Graph Learning for Molecular Property Prediction
- [AAAI 2021] ([paper](https://arxiv.org/pdf/2012.04915)) Progressive Network Grafting for Few-Shot Knowledge Distillation
