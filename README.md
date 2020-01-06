Awesome Papers Few-shot focus on collecting paper published on top conferences in Few-shot learning area,
hoping that this cut some time costing for beginners. Morever we also glad to see this repo can be a virtual online seminar,
which can be a home to all researchers who have the enthusiasm to exchange interesting ideas.

Awesome Papers Few-shot 收集了近几年定会中与小样本学习相关的论文，并简单的进行了类别划分与整理。
一方面，我们希望这个仓库能够帮助广大希望入坑小样本学习的同胞减少入坑搜集论文的时间成本。另一方面，
我们也希望这里能称为研究小样本学习的同胞们互相交流有趣想法的一个小平台。

The papers collected in this repo are manually selected by myself, I am hoping that more researchers interested in this area can maintain this repo together.

仓库中收藏的论文均为我本人从历年顶会中手动挑选并阅读过和小样本学习相关的论文，也希望能有广大的同行来共同维护它。
（注意：部分深入解释 Meta-Learning 的论文并未收入到此仓库中，有兴趣的朋友可以发 issue 一起讨论）。


### Contents
<!-- vim-markdown-toc GitLab -->

  * [Paper Collection Categories](#paper-collection-categories)
  * [Awesome Resources](#awesome-resources)
    * [Relevant Awesome Datasets Repo](#relevant-awesome-datasets-repo)
    * [Relevant Awesome Few-shot PlayGround Repo](#relevant-awesome-few-shot-playground-repo)
    * [Relevant Awesome Blogs](#relevant-awesome-blogs)
  * [How to recommend a paper](#how-to-recommend-a-paper)
* [Main Contributors](#main-contributors)

<!-- vim-markdown-toc -->


### Paper Collection Categories
The numbers of papers increased dramatically recently, so we decided to
separate papers into different files according to their categories. 
**Note that** if you don't find your researching categories below, give a shot into
the other categories. Basically, we sort thoses papers, which quantitiy is few than 5, to the other.

今年随着小样本相关的论文的大量出现，我们决定把这些论文分类成以下几种类别，并存在相应的文件夹内。
**需要注意的是**如果你没有发现研究的小样本方向的话，尝试在 Other
类别中找找，我们把出现的论文数少于 5 篇的方向暂时都划分为 Other 类。

<<<<<<< HEAD
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
- ref to this [url](https://github.com/Duan-JM/awesome-papers-fewshot/blob/master/DETECTION.md)

### Segementation
- [CVPR 2019] CANet- Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning
- [AAAI 2019] ([paper](https://www.researchgate.net/publication/335296764_Attention-Based_Multi-Context_Guiding_for_Few-Shot_Semantic_Segmentation)) Attention-based Multi-Context Guiding for Few-Shot Semantic Segmentation
    * Utilize the output of the different layers between query branch and support branch to gain more context informations.

- [arXiv 2019] AMP-Adaptive Masked Proxies for Few-Shot Segmentation
    * Not sure result in this area.
- [AAAI 2019] Unsupervised Meta-learning of Figure-Ground Segmentation via Imitating Visual Effects
    * Differetiate the background from images. 
=======
- Image Classification [*Jump here*](https://github.com/Duan-JM/awesome-papers-fewshot/blob/master/image_classification/README.md)
- Object Detection [*Jump here*](https://github.com/Duan-JM/awesome-papers-fewshot/blob/master/object_detection/README.md)
- Segementation [*Jump here*](https://github.com/Duan-JM/awesome-papers-fewshot/blob/master/segementation/README.md)
- Generators [*Jump here*](https://github.com/Duan-JM/awesome-papers-fewshot/blob/master/generators/README.md)
- Others [*Jump here*](https://github.com/Duan-JM/awesome-papers-fewshot/blob/master/others/README.md)
>>>>>>> upstream/master

### Awesome Resources
We collect some awesome code and blogs here.
**Note that** if you are now writing a few-shot papers, feel free to checkout `resources` folder under each categories to get some bib there

除了论文我们还在这里收藏了一些很棒的开源代码和博客。除此之外，如果您已经开始写论文的话，bib
引用相关文件可以在对应的文件夹中找到，希望这些能节省一部分您的写作时间。


#### Relevant Awesome Datasets Repo
- [pytorch-meta](https://github.com/tristandeleu/pytorch-meta) (Recommended)
- [meta-dataset](https://github.com/google-research/meta-dataset) (Received in ICLR 2020)
- [Few-Shot-Object-Detection-Dataset](https://github.com/fanq15/Few-Shot-Object-Detection-Dataset)


#### Relevant Awesome Few-shot PlayGround Repo
- [pytorch_metric_learning](https://github.com/KevinMusgrave/pytorch_metric_learning)


#### Relevant Awesome Blogs
- [Papers of Meta-Learning](https://github.com/sudharsan13296/Awesome-Meta-Learning)
- [Meta-Learning: Learning to Learn Fast](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)
- [大数据时代的小样本深度学习问题的综述](https://zhuanlan.zhihu.com/p/60881968)(Recommended)
- [Hands-On-Meta-Learning-With-Python](https://github.com/sudharsan13296/Hands-On-Meta-Learning-With-Python)


### How to recommend a paper
You are highly welcome to recommend a paper to this repo. 
The only thing you need to do is make a new issue with its name, conference name, years and some recommends words(no more than 400 words).

非常欢迎大家来推荐相关论文呀，推荐论文的方式非常简单，只需要提交一个 Issue，并在 Issue 中写清楚论文的题目，发表的会议名称以及年份和一个不超过 400 字的推荐理由即可。

> EXAMPLE
>
> Title: [ICML 2019] TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning
>
> Recommend: First paper point out how to measure the backbone is bad or good for the current task(episode).

## Main Contributors
- [Duan-JM](https://github.com/Duan-JM) (Founder) (Image Classification)
- [Bryce1010](https://github.com/Bryce1010) (Segementation)
- [ximingxing](https://github.com/ximingxing)
