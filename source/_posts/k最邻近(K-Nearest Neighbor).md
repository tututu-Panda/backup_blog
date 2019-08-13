---
title: K最邻近（K-Nearest—Neighbor）
tags: [KNN]
toc: true
grammar_cjkRuby: true
grammar_highlight: true
mathjax: true
---

## KNN思想

> KNN是什么?

K-Nearest-Neighbor,翻译过来就是k个最近的邻居。通过选取的k个最近的邻居，读取他们已知的结果，便可以根据”多数表决“的方式，将需要分类的数据进行判别。

<!--more-->

> 算法步骤描述

1. 计算待测数据与所有训练数据之间的距离。
2. 根据距离关系将其进行递增排序。
3. 选取k个最近的训练数据。
4. 统计该k个训练数据所属的类别，将最多的类别赋予待测数据。

## 算法实现

1. 数据创建
```python
# 导入数据
def file2matrix(filename):
    """
    :param filename: 数据文件路径
    :return: 数据矩阵 和 对应的类别标签
    """
    # 获取文件指针
    fr = open(filename)
    # 读取文件行数
    numberOfLine = len(fr.readlines())
    # 创建类别矩阵
    lableVector = []
    # 创建n行3列的矩阵
    returnMat = py.zeros((numberOfLine, 3))
    index = 0
    # ### 将文件指针重置
    fr = open(filename)
    # 遍历数据将其存入矩阵中
    for line in fr.readlines():
        # 剔除首尾空白
        line = line.strip()
        # 读取每一列数据
        listFromLine = line.split("\t")
        # 将数据映射到矩阵每一行中
        returnMat[index, :] = listFromLine[0:3]
        # 类别映射到矩阵中
        lableVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, lableVector
```

2. 归一化处理:将所有特征之间的**权重大小进行分配**，使之可以合理的参与最终的分类决策中去。
```python
def autoNorm(dataSet):
    """
    求出归一化后的数据
    :param dataSet: 数据集
    :return: 归一化的数据集normDataSet 范围ranges和最小值minVals
    归一化公式：
    Y = (x - xMin) / (xMax - xMin)
    """
    # 获取每列特征的最小值
    minVals = dataSet.min(0)
    # 获取每列特征的最大值
    maxVals = dataSet.max(0)
    # 求出特征值范围大小
    ranges = maxVals - minVals
    # 初始化数据集
    normDataSet = py.zeros(py.shape(dataSet))
    # 获取行数
    m = dataSet.shape[0]
    # 归一化数据集        py.tile(A, reps) => 将数据A重复rep次
    normDataSet = (dataSet - py.tile(minVals, (m, 1))) / py.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
```


3. 进行待测数据的分类
```python
def classify0(inX, dataSet, labels, k):
    """
    :param inX: 归一化距离
    :param dataSet:
    :param labels: 分类局怎
    :param k: 所需的分类对照个数
    :return:
    """
    dataSetSize = dataSet.shape[0]
    # 根据归一化距离，求出欧式距离度量
    diffMat = py.tile(inX, (dataSetSize, 1)) - dataSet
    # 距离的平方
    sqDiffMat = diffMat ** 2
    # 将每列相加（二维数组中0代表行，1代表列）
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    
    # 将距离从小到大排序
    sortedDistIndicies = distances.argsort()
    classCount = {}	# 训练数据的分类字典
    # 获取前k个距离，选取k个类型中最多的类别分类属性
    for i in range(k):
        # 获取标签
        voteLabel = labels[sortedDistIndicies[i]]	# 获取分类标签
        # 标签对应的值加1，默认为0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 将分类数据根据第二个元素进行排序，即根据分类的数据个数
    """
    a = [1,2,3]
    b = [[1,2,3], [4,5,6], [7,8,9]]
    operator.itemgetter(1) a=>[2] b=>[4,5,6]
    operator.itemgetter(2,1) a=>[3,2] b=>[[7,8,9],[4,5,6]]
    operator.itemgetter()相当于获取某个下标位置的元素
    """
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```

4. 主方法测试
```python
def classifyPerson():
    # 1. 进行标签类别标注
    resultList = ['not at all', 'in small doses', 'in large doses']
    # 2. 加载训练数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # 3. 进行属性归一化处理
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 4. 输入判定属性(每年飞行公里,看电影花的时间百分比,每年消耗的冰淇淋多少升)
    inArr = py.array([10000, 10, 0.5])
    # 5. 选取3个最邻近待测数据,得到分类标签
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])
```


## 问题分析

> 归一化的重要意义

如若存在多种属性，来判别是否为男性。**如身高，脚码等**。
假如有以下数据：[179,42,男]，[178,43,男],[165,36,女],[177,42,男],[160,35,女]。

通过观察可知，**身高的数据是脚码数据的4倍**。那么在进行距离度量时，则**更容易偏向身高的数据**，而脚码的数据则基本不足以影响判别。

如待测样本[176,43]与各训练数据之间的距离:
$$
\begin{array}{l}{A F=\sqrt{(167-179)^{2}+(43-42)^{2}}=\sqrt{145}} \\ {B F=\sqrt{(167-178)^{2}+(43-43)^{2}}=\sqrt{121}} \\ {C F=\sqrt{(167-165)^{2}+(43-36)^{2}}=\sqrt{101}} \\ {D F=\sqrt{(167-177)^{2}+(43-42)^{2}}=\sqrt{101}} \\ {E F=\sqrt{(167-160)^{2}+(43-35)^{2}}=\sqrt{103}}\end{array}
$$

取K为3时，为2女1男。

其被判定为女性。拥有43脚码且身高为176的女性？很少存在吧，一般都是男性拥有这种身材。因此其判定大概率出现错误。



因此归一化的重要意义便体现了出来:

**它可以预先调整数据，使得特征属性在数值之间的概率分布更加对齐，而不存在偏向某个特征属性的情况。也就是所谓的权重分配。**


> k的选取 

k的选取至关重要。

如果选取的数值太小如1，则选取的临近值无法具有说服力。而如果选取的数值太大如整体数据的3/4，那么则选取的临近值更倾向于整体数据的比重。

因此k的选取需要进行最优选择。最常用的为**交叉验证法（一部分做测试集，一部分做数据集）**

## 参考链接

[最近邻居法](https://zh.wikipedia.org/wiki/%E6%9C%80%E8%BF%91%E9%84%B0%E5%B1%85%E6%B3%95)
[KNN(K-Nearest Neighbor)分类算法原理](https://blog.csdn.net/shenziheng1/article/details/71891126)
[第2章 k-近邻算法](https://github.com/apachecn/AiLearning/blob/master/docs/ml/2.k-%E8%BF%91%E9%82%BB%E7%AE%97%E6%B3%95.md)
[什么是归一化](https://www.zhihu.com/question/19951858)