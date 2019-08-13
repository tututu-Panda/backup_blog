---
title: 决策树（id3算法）
tags: [决策树]
toc: true
grammar_cjkRuby: true
grammar_highlight: true
mathjax: true

---


## 概念

> 什么是决策树

决策树是对分类的一种方式。其是将问题的特征进行划分，并通过从上到下依次对比已有的数据特征，最终将问题划分为某个已知类别。

<!--more-->
> 决策树的相关概念

熵：表示随机事件的不确定性。

香农熵：指信息的不确定性的度量。信息越不确定，该信息熵也就越大。

香农熵公式：假如一个随机变量 X 的取值为 X={*x*1,*x*1,...,*x*n}，每一种取到的概率为：{*p*1,*p*1,...,pn}，那么X的熵定义则为：
$$
H(x)=-\sum_{i=1}^{n} p_{i} \log _{2} p_{i}
$$


条件熵：在一个条件下，随机变量的不确定性。

信息增益：熵 - 条件熵。 **即在一个条件下，信息不确定性减少的程度。**

> 常用结构

根节点、内部节点：特征标签

叶子节点：所属类别

例如一个邮件分类系统，可以通过决策树表示为：
![邮件分类](/img/决策树-流程图.jpg)





##  决策树构造算法

#### 1. ID3算法

> 划分依据

id3算法以信息论为基础，对每个特征标签的**信息熵和信息增益为判定条件**，来实现树的构造。

> 判断划分结束

有两种判定划分结束的情况：

1、 特征标签已经划分完了，无法继续划分。

2、划分出来的结果都属于同一个类。


> 具体实现步骤

主要通过递归的方式，构造决策树。
![id3流程步骤](/img/20140605135958281.jpg)


## 代码实现

1. 构造数据
```python
def createDataSet():
    """
    创建数据集以及标签
    :return:
    """
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
```

2. 计算香农熵

主要根据公式进行熵的计算：
$$
H(x)=-\sum_{i=1}^{n} p_{i} \log _{2} p_{i}
$$


```python
def calcShannonEnt(dataSet):
    """
    计算数据集的熵
    :param dataSet: 给定的数据集
    :return: 熵
    """
    numEntries = len(dataSet)
    labelCount = {}
    # 统计类别总数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCount[currentLabel] = labelCount.get(currentLabel, 0) + 1
    # 根据标签的占比，求出熵
    shannonEnt = 0.0
    for key in labelCount:
        # 类别出现频率
        prob = float(labelCount[key]) / numEntries
        # 计算熵，以2为底数求对数
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt
```


3. 根据特征标签划分数据
```python
def splitDataSet(dataSet, index, value):
    """
    按照给定的特征对数据集进行划分
    :param dataSet: 数据集
    :param index: 划分特征
    :param value: 返回特征的值
    :return:
    """
    returnDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        # 如果当前列特征的值等于value
        if featVec[index] == value:
            # 获取当前列之前列的特征
            reduceFeatVec = featVec[:index]
            # 剔除特征列
            reduceFeatVec.extend(featVec[index+1:])
            returnDataSet.append(reduceFeatVec)
    return returnDataSet
```

4. 获取划分的最适特征标签
```python
def chooseBestFeatureToSplit(dataSet):
    # 获取特征数量
    numFeatures = len(dataSet)
    # 获取香农熵
    bestEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值，以及最优的feature编号
    bestInfoGain, bestFeature = 0.0, -1
    # 遍历每一个特征列
    for i in range(numFeatures):
        # 获取每一列特征的所有数据
        featList = [example[i] for example in dataSet]
        # 得到所有分类
        uniqueVals = set(featList)
        # 临时信息熵
        newEntroy = 0.0
        # 遍历当前特征中的所有唯一属性值，对每一个唯一值进行数据划分，计算数据集的新熵值，并对所有唯一特征得到的熵求和
        for value in uniqueVals:
            # 根据属性值与特征列对数据集进行划分
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算出划分的数据集概率
            prob = len(subDataSet) / float(len(dataSet))
            # 求出划分的数据集熵
            newEntroy += prob * calcShannonEnt(subDataSet)
        # 获取划分数据集后信息熵的变化
        infoGain = bestEntropy - newEntroy
        print("infoGain=", infoGain, ",bestFeature = ", infoGain, "i = ", i, "bestEntropy = ", bestEntropy, " newEntroy = ", newEntroy)

        # 获取最好信息熵划分
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
        return bestFeature
```

5. 当标签类别只剩下一个时
```python
def majorityCnt(classList):
    """
    特征标签只剩下1个，选择出现次数最多的一个结果
    :param classList: 特征标签
    :return: 最优特征集
    """
    classCount={}
    # 统计类别结果个数
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
        # if vote not in classCount.keys(): classCount[vote] = 0
        # classCount[vote] += 1
    # 进行类别从高到低的排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```

6. 创建决策树
```python
def createTree(dataSet, labels):
    # 获取类别列表
    classList = [example[-1] for example in dataSet]
    # 结束条件1：如果只有一个类别，则直接返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 结束条件2：特征标签只剩下1个
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 获取最佳分类特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 得到最好分类特征标签
    bestFeatLabel = labels[bestFeat]
    # 构建决策树
    myTree = {bestFeatLabel:{}}
    # 将最佳特征标签从标签中删除
    del(labels[bestFeat])
    # 得到最佳特征的所有唯一数据
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 依据最佳特征，对数据进行递归划分
    for value in uniqueVals:
    	# 获取剩余特征标签
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
```

7. 主方法调用
```python
dataSet, labels = createDataSet()
import copy
myTree = createTree(dataSet, copy.deepcopy(labels))
print(myTree)
```

## 代码分析

1. 主要通过6去创建决策树，首先遍历出所有的类别信息，判断是否需要结束。
2. 根据现有的数据集，找出最佳的分类特征 =》 4
3. 计算出当前数据未划分时的香农熵 =》2
4. 遍历每一个特征标签，根据特征标签的值进行划分，求出划分后的香农熵，根据划分后的香浓熵，求出信息增益
5. 得到最大收益的特征标签，及当前的最佳划分依据
6. 根据最佳划分依据进行剩余特征标签的递归划分 =》 1

## 参考链接

[第3章 决策树](https://github.com/apachecn/AiLearning/blob/master/docs/ml/3.%E5%86%B3%E7%AD%96%E6%A0%91.md) 
[信息熵是什么？](https://www.zhihu.com/question/22178202)
[信息增益到底怎么理解呢？](https://www.zhihu.com/question/22104055)
[简单易学的机器学习算法——决策树之ID3算法](https://blog.csdn.net/google19890102/article/details/28611225)