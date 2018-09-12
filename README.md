Squeeze-and-Excitation Networks
==============================

# 一. SE Net 简介

> SE Net是由Momenta公司提出并发于2017CVPR，论文的核心点在对CNN中的feature channel（特征通道依赖性）利用和创新。Sequeeze-and-Excitation(SE) block并不是一个完整的网络结构，而是一个子结构，可以嵌到其他分类或检测模型中，作者采用SENet block和ResNeXt结合在ILSVRC 2017的分类项目中拿到第一，在ImageNet数据集上将top-5 error降低到2.251%，原先的最好成绩是2.991%。作者在文中将SENet block插入到现有的多种分类网络中，都取得了不错的效果。SENet的核心思想在于通过网络根据loss去学习特征权重，使得有效的feature map权重大，无效或效果小的feature map权重小的方式训练模型达到更好的结果。当然，SE block嵌在原有的一些分类网络中不可避免地增加了一些参数和计算量，但是在效果面前还是可以接受的。

# 二. SE Net详解

> SE Net的核心设计是SE building block——Squeeze和Excitation，如图所示：

imag111111111

## (一) <img src="https://latex.codecogs.com/svg.latex?\LARGE&space;F_{tr}" title="\LARGE F_{tr}" />

> X——>U的实现过程是<img src="https://latex.codecogs.com/svg.latex?F_{tr}" title="F_{tr}" />，一般就是CNN中的一些普通的操作，例如卷积或一组卷积。输入输出定义如下：

images11111

## (二) Squeeze: Global InformationEmbedding

> Squeeze操作就是在得到U（多个feature map）之后采用全局平均池化操作对其每个feature map进行压缩，使其C个feature map最后变成1 x 1 x C的实数数列，如公式所示：

img11111111

> 一般CNN中的每个通道学习到的滤波器都对局部感受野进行操作，因此U中每个feature map都无法利用其它feature map的上下文信息，而且网络较低的层次上其感受野尺寸都是很小的，这样情况就会更严重。 U（多个feature map）可以被解释为局部描述的子集合，这些描述的统计信息对于整个图像来说是有表现力的。论文选择最简单的全局平均池化操作，从而使其具有全局的感受野，使得网络低层也能利用全局信息。

## (三) Excitation: Adaptive Recalibration

> 接下来就是Excitation操作，该过程为两个全连接操作，如公式所示：



### 1. 全连接层1
> 公式如下：<img src="https://latex.codecogs.com/svg.latex?\delta&space;(W_{1}z)" title="\delta (W_{1}z)" />

> squeeze得到的结果是z，z的维度是1 x 1 x C；W1的维度是C/r * C，其中r是缩放参数，在文中取的是16，这个参数的目的是为了减少channel个数从而降低计算量；<img src="https://latex.codecogs.com/svg.latex?W_{1}z" title="W_{1}z" />的维度为1 x 1 x C/r，之后经过一个ReLU层，输出的维度不变。

### 2. 全连接层2
> 公式如下：<img src="https://latex.codecogs.com/svg.latex?\sigma&space;(W{_2}\delta(W_{1}z))" title="\sigma (W{_2}\delta(W_{1}z))" />

> 全连接层1的结果和W2相乘，W2的维度是C x C/r，输出的维度是1 x 1 x C；最后再经过sigmoid函数，得到s，s的维度是1 x 1 x C，C表示channel数目。s是本文的核心，它是用来刻画tensor U中C个feature map的权重。而且这个权重是通过前面这些全连接层和非线性层学习得到的，因此可以end-to-end训练。这两个全连接层的作用就是融合各通道的feature map信息，因为前面的squeeze都是在某个channel的feature map里面操作。

## (四) Scale

> scale公式如下：
imag111

> 得到s之后，就可以对原来的tensor U做channel-wise multiplication操作——uc是一个二维矩阵，sc是一个数，也就是权重，因此相当于把uc矩阵中的每个值都乘以sc，该过程对应图中的Fscale。

# 三. SE Net应用

> 将SE Net添加进Inception和ResNet中，SE building block结构如图所示：

imag111111

> SE-ResNet-50和SE-ResNeXt-50具体结构如图所示：

imag1111111

# 四. 代码

> 利用MNIST数据集，构建SE-Resnet50网络，查看网络效果，由于输入为28 x 28，所以最后的全局池化没有用到，代码如下：

