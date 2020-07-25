<!-- vim-markdown-toc GitLab -->

* [Domain Adaptation](#domain-adaptation)
* [Reinforcement Learning](#reinforcement-learning)
* [Visual Tracking](#visual-tracking)
* [NLP relevant](#nlp-relevant)
  * [Representation](#representation)
  * [NLU](#nlu)
  * [DCM](#dcm)
  * [NLG](#nlg)
  * [Applications](#applications)
* [Theoritical](#theoritical)
* [Relation Relevant](#relation-relevant)
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
- [ICML 2020] Few-shot Domain Adaptation by Causal Mechanism Transfer

### Reinforcement Learning
- [ICML 2019] Few-Shot Intent Inference via Meta-Inverse Reinforcement Learning
- [ICLR 2020] Meta Reinforcement Learning with Autonomous Inference of Subtask Dependencies
- [ICLR 2020] VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning
- [ICLR 2020 Bengio] A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms

### Visual Tracking
- [ICCV 2019] Deep Meta Learning for Real-Time Target-Aware Visual Tracking
- [CVPR 2020] ([paper](https://arxiv.org/abs/2004.00830)) Tracking by Instance Detection: A Meta-Learning Approach
    * MAML-Tracker

### NLP relevant
#### Representation
- [ACL 2020] Shaping Visual Representations with Language for Few-shot Classification
    * jointly predicting natural language task descriptions at training time
    * How can we let language guide representa- tion learning in machine learning models? 
- [arXiv 2020] Language Models are Few-Shot Learners
    * GPT-3
    * add here for its
- [EMNLP 2019] ([paper](https://arxiv.org/pdf/1902.10482.pdf)) Few-Shot Text Classification with Induction Network
    * Introduce dynamic routing to generate better class representations. One real industrial project.
- [ACL 2020] Span-ConveRT: Few-shot Span Extraction for Dialog with Pretrained Conversational Representations

#### NLU
- [ACL 2020] Dynamic Memory Induction Networks for Few-Shot Text Classification
    * result seems great
    * work follow the inductino network, explicitly add dynamic memory model (constructed by base classes) to enhance "prototypes".
- [ICLR 2020] Few-shot Text Classification with Distributional Signatures
- [EMNLP/IJCNLP 2019] Hierarchical Attention Prototypical Networks for Few-Shot Text Classification

#### DCM
- [ACL 2020] Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network

#### NLG
- [IJCAI 2019] Meta-Learning for Low-resource Natural Language Generation in Task-oriented Dialogue Systems
- [ACL 2020] Few-Shot NLG with Pre-Trained Language Model
- [ACL 2020] Learning to Customize Model Structures for Few-shot Dialogue Generation Tasks

#### Applications
- [AAAI 2018] Few Shot Transfer Learning Between Word Relatedness and Similarity Tasks Using A Gated Recurrent Siamese Network
- [ACMMM 2018] Few-Shot Adaptation for Multimedia Semantic Indexing
- [ACMMM 2018] Fast Parameter Adaptation for Few-shot Image Captioning and Visual Question Answering
- [AAAI 2019] Few-Shot Image and Sentence Matching via Gated Visual-Semantic Embedding
    * Image and Sentence Matching
- [ACL 2020] Hypernymy Detection for Low-Resource Languages via Meta Learning
- [ICLR 2020] FEW-SHOT LEARNING ON GRAPHS VIA SUPERCLASSES BASED ON GRAPH SPECTRAL MEASURES
- [EMNLP 2019] Meta Relational Learning for Few-Shot Link Prediction in Knowledge Graphs
- [EMNLP 2019] Adapting Meta Knowledge Graph Information for Multi-Hop Reasoning over Few-Shot Relations
- [EMNLP 2019] FewRel 2.0: Towards More Challenging Few-Shot Relation Classification

### Theoritical
- [ICLR 2020 Bengio] A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms
    * Argue that how to fast adapt to new distributions by meta-learn causal structures
    * Also have follow paper on arxiv [here](https://www.semanticscholar.org/paper/An-Analysis-of-the-Adaptation-Speed-of-Causal-Priol-Harikandeh/982816b03c2f09f7eb63f40dfbedd03caa3e5570?utm_source=alert_email&utm_content=AuthorPaper&utm_campaign=AlertEmails_DAILY&utm_term=AuthorPaper&email_index=0-0-0&utm_medium=244646)

### Relation Relevant
- [ICML 2020] ([paper](https://arxiv.org/abs/2007.02387)) Few-shot Relation Extraction via Bayesian Meta-learning on Task Graph
- [AAAI 2019] Hybrid Attention-based Prototypical Networks for Noisy Few-Shot Relation Classification
    * Relation Classification with FewRel
- [AAAI 2020] Neural Snowball for Few-Shot Relation Learning

### Others
- [IJCAI 2019] Incremental Few-Shot Learning for Pedestrian Attribute Recognition
- [AAAI 2018] AffinityNet- Semi-supervised Few-shot Learning for Disease Type Prediction
    * Use few-shot method to enhance oringal disease type prediction


- [NIPS 2018] Neural Voice Cloning with a Few Samples
- [ICCV 2019] ([paper](https://arxiv.org/pdf/1909.01205)) Few-Shot Generalization for Single-Image 3D Reconstruction via Priors
- [ICCV 2019] ([paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_ACMM_Aligned_Cross-Modal_Memory_for_Few-Shot_Image_and_Sentence_Matching_ICCV_2019_paper.pdf)) ACMM: Aligned Cross-Modal Memory for Few-Shot Image and Sentence Matching
    * Image and Sentence Matching
- [ICCV 2019] (RECOMMANDED!) Task-Driven Modular Networks for Zero-Shot Compositional Learning
    * An interesting usage of a bunch of MLPs.
- [CVPR 2019] ([paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Rostami_SAR_Image_Classification_Using_Few-Shot_Cross-Domain_Transfer_Learning_CVPRW_2019_paper.pdf) [code](https://github.com/MSiam/AdaptiveMaskedProxies.)) SAR Image Classification Using Few-shot Cross-domain Transfer Learning
- [ICLR 2020] FEW-SHOT TEXT CLASSIFICATION WITH DISTRIBUTIONAL SIGNATURES
- [ICLR 2020] METAPIX: FEW-SHOT VIDEO RETARGETING
- [ICLR 2020] ONE-SHOT PRUNING OF RECURRENT NEURAL NETWORKS BY JACOBIAN SPECTRUM EVALUATION
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
- [AAAI 2020] Few-Shot Knowledge Graph Completion (关系抽取)
- [AAAI 2020] Learning Meta Model for Zero- and Few-shot Face Anti-spoofing
- [AAAI 2020] Graph Few-shot Learning via Knowledge Transfer
- [AAAI 2020] Few Shot Network Compression via Cross Distillation (模型压缩)
- [AAAI 2020] Few-Shot Bayesian Imitation Learning with Logical Program Policies
- [CVPR 2020] Meta-Transfer Learning for Zero-Shot Super-Resolution
- [CVPR 2020] Learning from Web Data with Self-Organizing Memory Module
    * solve label noise and background noise in the images with memory module
- [CVPR 2020] Single-view view synthesis with multiplane images
- [AAAI 2020] ([paper](https://aaai.org/Papers/AAAI/2020GB/AAAI-JiZ.4799.pdf)) SGAP-Net: Semantic-Guided Attentive Prototypes Network for Few-Shot Human-Object Interaction Recognition
- [ICLR 2020] MetaPix: Few-Shot Video Retargeting
- [ICLR 2020] ([paper](https://openreview.net/forum?id=r1eowANFvr)) Towards Fast Adaptation of Neural Architectures with Meta Learning 
- [ICLR 2020] Query-efficient Meta Attack to Deep Neural Networks
- [SIGGRAPH Asia 2019] Artistic Glyph Image Synthesis via One-Stage Few-Shot Learning
- [IJCNN 2020] Interpretable Time-series Classification on Few-shot Samples
- [CVPR 2018] ([paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Temporal_Hallucinating_for_CVPR_2018_paper.pdf)) Temporal Hallucinating for Action Recognition with Few Still Images
    * Attempt to recall cues from relevant action videos.
    * Maybe good at one-shot, not worse than the baseline in 5-shot and 10-shot scenarios.
