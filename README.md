## Awesome Papers - Few shot 

![](https://img.shields.io/badge/FewShot-study-yellowgreen)
![](https://img.shields.io/badge/Welcome-contributors-yellowbrightgreen)

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
  * [ChangeLog](#changelog)
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

- Image Classification [*Jump here*](https://github.com/Duan-JM/awesome-papers-fewshot/blob/master/image_classification/README.md)
- Object Detection [*Jump here*](https://github.com/Duan-JM/awesome-papers-fewshot/blob/master/object_detection/README.md)
- Segementation [*Jump here*](https://github.com/Duan-JM/awesome-papers-fewshot/blob/master/segementation/README.md)
- Generators [*Jump here*](https://github.com/Duan-JM/awesome-papers-fewshot/blob/master/generators/README.md)
- Others [*Jump here*](https://github.com/Duan-JM/awesome-papers-fewshot/blob/master/others/README.md)

### Awesome Resources
We collect some awesome code and blogs here.
**Note that** if you are now writing a few-shot papers, feel free to checkout `resources` folder under each categories to get some bib there

除了论文我们还在这里收藏了一些很棒的开源代码和博客。除此之外，如果您已经开始写论文的话，bib
引用相关文件可以在对应的文件夹中找到，希望这些能节省一部分您的写作时间。


#### Relevant Awesome Datasets Repo
- [pytorch-meta](https://github.com/tristandeleu/pytorch-meta) (Recommended)
- [meta-dataset](https://github.com/google-research/meta-dataset) (Received in ICLR 2020)
- [Few-Shot-Object-Detection-Dataset](https://github.com/fanq15/Few-Shot-Object-Detection-Dataset)
- [Few-shot-Segmentations-1000](https://github.com/HKUSTCV/FSS-1000)
- [Northumberland Dolphin Dataset 2020](https://doi.org/10.25405/data.ncl.c.4982342) (CVPR 2020 workshop)
- [mini-ImageNet contained label embeddings](https://drive.google.com/file/d/1g4wOa0FpWalffXJMN2IZw0K2TM2uxzbk/view) (Used in AM3)
- [tiered-ImageNet contained label embeddings](https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view) (Used in AM3)
- [FewRel](https://github.com/thunlp/FewRel) (EMNLP 2018)


#### Relevant Awesome Few-shot PlayGround Repo
- [pytorch_metric_learning](https://github.com/KevinMusgrave/pytorch_metric_learning)


#### Relevant Awesome Blogs
- [Papers of Meta-Learning](https://github.com/sudharsan13296/Awesome-Meta-Learning)
- [Meta-Learning: Learning to Learn Fast](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)
- [大数据时代的小样本深度学习问题的综述](https://zhuanlan.zhihu.com/p/60881968)(Recommended)
- [Hands-On-Meta-Learning-With-Python](https://github.com/sudharsan13296/Hands-On-Meta-Learning-With-Python)
- [Floodsung's Meta-Learning Paper lists](https://github.com/floodsung/Meta-Learning-Papers)
- [Open-Sourcing BiT: Exploring Large-Scale Pre-training for Computer Vision](https://ai.googleblog.com/2020/05/open-sourcing-bit-exploring-large-scale.html?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+blogspot%2FgJZg+%28Google+AI+Blog%29)
    * Large-scale pretrained strategy can be used to solve low resource circumstances.
- [Meta-Learning in Neural Netorks: A Survey](https://arxiv.org/pdf/2004.05439.pdf)
    * Meta-Learning methods are not equal to Few-shot Learning methods.

### How to recommend a paper
You are highly welcome to recommend a paper to this repo. 
The only thing you need to do is make a new issue with its name, conference name, years and some recommends words(no more than 400 words).

非常欢迎大家来推荐相关论文呀，推荐论文的方式非常简单，只需要提交一个 Issue，并在 Issue 中写清楚论文的题目，发表的会议名称以及年份和一个不超过 400 字的推荐理由即可。

> EXAMPLE
>
> Title: [ICML 2019] TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning
>
> Recommend: First paper point out how to measure the backbone is bad or good for the current task(episode).

### ChangeLog
- 2020-06-26 `REAMIN_SORTED_PAPAERS.md` no longer collected all papers from arxiv, only interesting and effectiveness ones will be collected here

## Main Contributors
- [Duan-JM](https://github.com/Duan-JM) (Founder) (Image Classification)
- [Bryce1010](https://github.com/Bryce1010) (Segementation)
- [ximingxing](https://github.com/ximingxing)
