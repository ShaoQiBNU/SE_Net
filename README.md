Squeeze-and-Excitation Networks
==============================

# 一. SE Net 简介

> SE Net是由Momenta公司提出并发于2017CVPR，论文的核心点在对CNN中的feature channel（特征通道依赖性）利用和创新。Sequeeze-and-Excitation(SE) block并不是一个完整的网络结构，而是一个子结构，可以嵌到其他分类或检测模型中，作者采用SENet block和ResNeXt结合在ILSVRC 2017的分类项目中拿到第一，在ImageNet数据集上将top-5 error降低到2.251%，原先的最好成绩是2.991%。作者在文中将SENet block插入到现有的多种分类网络中，都取得了不错的效果。SENet的核心思想在于通过网络根据loss去学习特征权重，使得有效的feature map权重大，无效或效果小的feature map权重小的方式训练模型达到更好的结果。当然，SE block嵌在原有的一些分类网络中不可避免地增加了一些参数和计算量，但是在效果面前还是可以接受的。

# 二. SE Net详解

> SE Net的核心设计是SE building block——Squeeze和Excitation，如图所示：

![image](https://github.com/ShaoQiBNU/SE_Net/blob/master/images/1.png)

## (一) <img src="https://latex.codecogs.com/svg.latex?\LARGE&space;F_{tr}" title="\LARGE F_{tr}" />

> X——>U的实现过程是<img src="https://latex.codecogs.com/svg.latex?F_{tr}" title="F_{tr}" />，一般就是CNN中的一些普通的操作，例如卷积或一组卷积。输入输出定义如下：

![image](https://github.com/ShaoQiBNU/SE_Net/blob/master/images/2.png)

## (二) Squeeze: Global InformationEmbedding

> Squeeze操作就是在得到U（多个feature map）之后采用全局平均池化操作对其每个feature map进行压缩，使其C个feature map最后变成1 x 1 x C的实数数列，如公式所示：

![image](https://github.com/ShaoQiBNU/SE_Net/blob/master/images/3.png)

> 一般CNN中的每个通道学习到的滤波器都对局部感受野进行操作，因此U中每个feature map都无法利用其它feature map的上下文信息，而且网络较低的层次上其感受野尺寸都是很小的，这样情况就会更严重。 U（多个feature map）可以被解释为局部描述的子集合，这些描述的统计信息对于整个图像来说是有表现力的。论文选择最简单的全局平均池化操作，从而使其具有全局的感受野，使得网络低层也能利用全局信息。

## (三) Excitation: Adaptive Recalibration

> 接下来就是Excitation操作，该过程为两个全连接操作，如公式所示：

![image](https://github.com/ShaoQiBNU/SE_Net/blob/master/images/4.png)

### 1. 全连接层1
> 公式如下：<img src="https://latex.codecogs.com/svg.latex?\delta&space;(W_{1}z)" title="\delta (W_{1}z)" />

> squeeze得到的结果是z，z的维度是1 x 1 x C；W1的维度是C/r * C，其中r是缩放参数，在文中取的是16，这个参数的目的是为了减少channel个数从而降低计算量；<img src="https://latex.codecogs.com/svg.latex?W_{1}z" title="W_{1}z" />的维度为1 x 1 x C/r，之后经过一个ReLU层，输出的维度不变。

### 2. 全连接层2
> 公式如下：<img src="https://latex.codecogs.com/svg.latex?\sigma&space;(W{_2}\delta(W_{1}z))" title="\sigma (W{_2}\delta(W_{1}z))" />

> 全连接层1的结果和W2相乘，W2的维度是C x C/r，输出的维度是1 x 1 x C；最后再经过sigmoid函数，得到s，s的维度是1 x 1 x C，C表示channel数目。s是本文的核心，它是用来刻画tensor U中C个feature map的权重。而且这个权重是通过前面这些全连接层和非线性层学习得到的，因此可以end-to-end训练。这两个全连接层的作用就是融合各通道的feature map信息，因为前面的squeeze都是在某个channel的feature map里面操作。

## (四) Scale

> scale公式如下：

![image](https://github.com/ShaoQiBNU/SE_Net/blob/master/images/5.png)

> 得到s之后，就可以对原来的tensor U做channel-wise multiplication操作——uc是一个二维矩阵，sc是一个数，也就是权重，因此相当于把uc矩阵中的每个值都乘以sc，该过程对应图中的Fscale。

# 三. SE Net应用

> 将SE Net添加进Inception和ResNet中，SE building block结构如图所示：

![image](https://github.com/ShaoQiBNU/SE_Net/blob/master/images/6.png)

> SE-ResNet-50和SE-ResNeXt-50具体结构如图所示：

![image](https://github.com/ShaoQiBNU/SE_Net/blob/master/images/7.png)

# 四. 代码

> 利用MNIST数据集，构建SE-Resnet50网络，查看网络效果，由于输入为28 x 28，所以最后的全局池化没有用到，代码如下：

```python

########## load packages ##########
import tensorflow as tf

##################### load data ##########################
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("mnist_sets",one_hot=True)

########## set net hyperparameters ##########
learning_rate=0.0001

epochs=2
batch_size_train=128
batch_size_test=100

display_step=20

r = 16

########## set net parameters ##########
#### img shape:28*28 ####
n_input=784 

#### 0-9 digits ####
n_classes=10

########## placeholder ##########
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])


##################### build net model ##########################

######### SE_block #########
def SE_block(inputs, C):

    _,k1,k2,_=inputs.shape

    global_pool = tf.nn.avg_pool(inputs, ksize=[1,k1,k2,1], strides=[1,k1,k2,1], padding='VALID')

    fc1 = tf.layers.dense(global_pool, int(C/r))

    fc1 = tf.nn.relu(fc1)

    fc2 = tf.layers.dense(fc1, C)

    out = tf.sigmoid(fc2)

    out = tf.reshape(out, [-1,1,1,C])

    return out

######### identity_block #########
def identity_block(inputs,filters,kernel,strides):
    '''
    identity_block: 三层的恒等残差块，影像输入输出的height和width保持不变，channel发生变化
    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长
    return: out 三层恒等残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1, f2, f3=filters
    k1, k2, k3=kernel
    s1, s2, s3=strides


    ######## shortcut 第一种规则，影像输入输出的height和width保持不变，输入直接加到卷积结果上 ########
    inputs_shortcut=inputs


    ######## first identity block 第一层恒等残差块 ########
    #### conv ####
    layer1=tf.layers.conv2d(inputs,filters=f1,kernel_size=k1,strides=s1,padding='SAME')

    #### BN ####
    layer1=tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1=tf.nn.relu(layer1)


    ######## second identity block 第二层恒等残差块 ########
    #### conv ####
    layer2=tf.layers.conv2d(layer1,filters=f2,kernel_size=k2,strides=s2,padding='SAME')

    #### BN ####
    layer2=tf.layers.batch_normalization(layer2)

    #### relu ####
    layer2=tf.nn.relu(layer2)


    ######## third identity block 第三层恒等残差块 ########
    #### conv ####
    layer3=tf.layers.conv2d(layer2,filters=f3,kernel_size=k3,strides=s3,padding='SAME')

    #### BN ####
    layer3=tf.layers.batch_normalization(layer3)

    ######## SE block SE模块得到scale，与layer3相乘 ########
    scale=SE_block(layer3, f3)

    scale_layer = scale * layer3

    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out=tf.add(inputs_shortcut,scale_layer)


    ######## relu ########
    out=tf.nn.relu(out)
    
    return out


######## convolutional_block #########
def convolutional_block(inputs,filters,kernel,strides):
    '''
    convolutional_block: 三层的卷积残差块，影像输入输出的height、width和channel均发生变化
    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长
    return: out 三层的卷积残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1, f2, f3=filters
    k1, k2, k3=kernel
    s1, s2, s3=strides


    ######## shortcut 第二种规则，影像输入输出height和width发生变化，需要对输入做调整 ########
    #### conv ####
    inputs_shortcut=tf.layers.conv2d(inputs,filters=f3,kernel_size=1,strides=s1,padding='SAME')

    #### BN ####
    inputs_shortcut=tf.layers.batch_normalization(inputs_shortcut)


    ######## first convolutional block 第一层卷积残差块 ########
    #### conv ####
    layer1=tf.layers.conv2d(inputs,filters=f1,kernel_size=k1,strides=s1,padding='SAME')

    #### BN ####
    layer1=tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1=tf.nn.relu(layer1)


    ######## second convolutional block 第二层卷积残差块 ########
    #### conv ####
    layer2=tf.layers.conv2d(layer1,filters=f2,kernel_size=k2,strides=s2,padding='SAME')

    #### BN ####
    layer2=tf.layers.batch_normalization(layer2)

        #### relu ####
    layer2=tf.nn.relu(layer2)


        ######## third convolutional block 第三层卷积残差块 ########
    #### conv ####
    layer3=tf.layers.conv2d(layer2,filters=f3,kernel_size=k3,strides=s3,padding='SAME')

    #### BN ####
    layer3=tf.layers.batch_normalization(layer3)

    ######## SE block SE模块得到scale，与layer3相乘 ########
    scale=SE_block(layer3, f3)

    scale_layer = scale * layer3

    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out=tf.add(inputs_shortcut,scale_layer)

    ######## relu ########
    out=tf.nn.relu(out)
    
    return out


######### Resnet 50 layer ##########
def Resnet50(x,n_classes):

    ####### reshape input picture ########
    x=tf.reshape(x,shape=[-1,28,28,1])


    ####### first conv ########
    #### conv ####
    conv1=tf.layers.conv2d(x,filters=64,kernel_size=7,strides=2,padding='SAME')

    #### BN ####
    conv1=tf.layers.batch_normalization(conv1)

    #### relu ####
    conv1=tf.nn.relu(conv1)


    ####### max pool ########
    pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


    ####### second conv ########
    #### convolutional_block 1 ####
    conv2=convolutional_block(pool1,filters=[64,64,256],kernel=[1,3,1],strides=[1,1,1])

    #### identity_block 2 ####
    conv2=identity_block(conv2,filters=[64,64,256],kernel=[1,3,1],strides=[1,1,1])
    conv2=identity_block(conv2,filters=[64,64,256],kernel=[1,3,1],strides=[1,1,1])


    ####### third conv ########
    #### convolutional_block 1 ####
    conv3=convolutional_block(conv2,filters=[128,128,512],kernel=[1,3,1],strides=[2,1,1])

    #### identity_block 3 ####
    conv3=identity_block(conv3,filters=[128,128,512],kernel=[1,3,1],strides=[1,1,1])
    conv3=identity_block(conv3,filters=[128,128,512],kernel=[1,3,1],strides=[1,1,1])
    conv3=identity_block(conv3,filters=[128,128,512],kernel=[1,3,1],strides=[1,1,1])


    ####### fourth conv ########
    #### convolutional_block 1 ####
    conv4=convolutional_block(conv3,filters=[256,256,1024],kernel=[1,3,1],strides=[2,1,1])
    
    #### identity_block 5 ####
    conv4=identity_block(conv4,filters=[256,256,1024],kernel=[1,3,1],strides=[1,1,1])
    conv4=identity_block(conv4,filters=[256,256,1024],kernel=[1,3,1],strides=[1,1,1])
    conv4=identity_block(conv4,filters=[256,256,1024],kernel=[1,3,1],strides=[1,1,1])
    conv4=identity_block(conv4,filters=[256,256,1024],kernel=[1,3,1],strides=[1,1,1])
    conv4=identity_block(conv4,filters=[256,256,1024],kernel=[1,3,1],strides=[1,1,1])


    ####### fifth conv ########
    #### convolutional_block 1 ####
    conv5=convolutional_block(conv4,filters=[512,512,2048],kernel=[1,3,1],strides=[2,1,1])
    
    #### identity_block 2 ####
    conv5=identity_block(conv5,filters=[512,512,2048],kernel=[1,3,1],strides=[1,1,1])
    conv5=identity_block(conv5,filters=[512,512,2048],kernel=[1,3,1],strides=[1,1,1])


    ####### 全局平均池化 ########
    #pool2=tf.nn.avg_pool(conv5,ksize=[1,7,7,1],strides=[1,7,7,1],padding='VALID')


    ####### flatten 影像展平 ########
    flatten = tf.reshape(conv5, (-1, 1*1*2048))


    ####### out 输出，10类 可根据数据集进行调整 ########
    out=tf.layers.dense(flatten,n_classes)


    ####### softmax ########
    out=tf.nn.softmax(out)

    return out


########## define model, loss and optimizer ##########

#### model pred 影像判断结果 ####
pred=Resnet50(x,n_classes)

#### loss 损失计算 ####
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


##################### train and evaluate model ##########################

########## initialize variables ##########
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step=1

    #### epoch 世代循环 ####
    for epoch in range(epochs+1):

        #### iteration ####
        for _ in range(mnist.train.num_examples//batch_size_train):

            step += 1

            ##### get x,y #####
            batch_x, batch_y=mnist.train.next_batch(batch_size_train)

            ##### optimizer ####
            sess.run(optimizer,feed_dict={x:batch_x, y:batch_y})

            
            ##### show loss and acc ##### 
            if step % display_step==0:
                loss,acc=sess.run([cost, accuracy],feed_dict={x: batch_x, y: batch_y})
                print("Epoch "+ str(epoch) + ", Minibatch Loss=" + \
                    "{:.6f}".format(loss) + ", Training Accuracy= "+ \
                    "{:.5f}".format(acc))


    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples//batch_size_test):
        batch_x,batch_y=mnist.test.next_batch(batch_size_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
```
