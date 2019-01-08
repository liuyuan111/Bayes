
# coding: utf-8

# In[1]:


# https://blog.csdn.net/moxigandashu/article/details/71480251?locationNum=16&fps=1
import numpy as np

# 构造loadDataSet函数用于生成实验样本
def loadDataSet(): 
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1] #1表示侮辱性言论，0表示正常言论
    return postingList,classVec

#构建词汇表生成函数creatVocabList
def createVocabList(dataSet):
    vocabSet=set([]) # 1 1 1 1 和1 1
    for document in dataSet:
        vocabSet=vocabSet|set(document) #取两个集合的并集
    return list(vocabSet)

# 对输入的词汇表构建词向量，
# 词集模型只记录了每个词是否出现，而没有记录词出现的次数，
# 如果在词向量中记录词出现的次数，每出现一次，则多记录一次，这样的词向量构建方法，被称为词袋模型
#词集模型
def setOfWords2Vec(vocabList,inputSet):
    returnVec=np.zeros(len(vocabList)) #生成零向量的array
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1 #有单词，则该位置填充0
        else: print('the word:%s is not in my Vocabulary!'% word)
    return returnVec #返回全为0和1的向量
#词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec #返回非负整数的词向量


listPosts,listClasses=loadDataSet()
myVocabList=createVocabList(listPosts)
print(myVocabList) # 输出词表
returnVec = setOfWords2Vec(myVocabList, listPosts[0])
print(returnVec) # 对输入的词汇表构建词向量，使用词集模型
returnVec = bagOfWords2VecMN(myVocabList, listPosts[0])
print(returnVec) # 对输入的词汇表构建词向量，使用词袋模型

# 运用词向量计算概率
def trainNB1(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWord=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/len(trainCategory) # 3/6
    p0Num=np.ones(numWord);p1Num=np.ones(numWord)# 初始化为1
    p0Demon=2;p1Demon=2 #初始化为2
    for i in range(numTrainDocs):
        if trainCategory[i]==0:
            p0Num+=trainMatrix[i]
            p0Demon+=sum(trainMatrix[i])
        else:
            p1Num+=trainMatrix[i]
            p1Demon+=sum(trainMatrix[i])
    p0Vec=np.log(p0Num/p0Demon) #对结果求自然对数
    p1Vec=np.log(p1Num/p1Demon) #对结果求自然对数
    return p0Vec,p1Vec,pAbusive

# 计算文档在各类中的概率，取较大者作为该文档的分类，所以构建分类函数classifyNB
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    # 说明： p1=sum(vec2Classify*p1Vec)+log(pClass1) 的数学原理是ln(a*b)=ln(a) +ln(b)
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1) 
    p0=sum(vec2Classify*p0Vec)+np.log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

# 构造几个样本，来测试分类函数
def testingNB():
    listPosts,listClasses=loadDataSet() # listPosts：feature， listClasses：label
    myVocabList=createVocabList(listPosts) # 转换 I am a Chinese  构造词典
    trainMat=[]
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc)) # 转换后的词表
    p0V,p1V,pAb=trainNB1(trainMat,listClasses) # 训练
    testEntry=['love','my','dalmation']
    thisDoc=setOfWords2Vec(myVocabList,testEntry)# 转换测试语句词表
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)) # 预测
    testEntry=['stupid','garbage']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

testingNB()

