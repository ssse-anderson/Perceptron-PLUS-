import numpy as np
from matplotlib import pyplot as plt 
 
def createDataSet():  
    global  group,labels
    group = np.array([[1,0], [0.2,-0.2], [1,1],[0,0],[0.8,-0.5],[0.5,0.9],[1.2,1],[0.75,0.66]])  #设矩阵
    labels = [-1, 1, -1,1,-1, 1, -1,1] #贴标签  

def perceptronClassify(trainGroup,trainLabels):
    global w, b

    n = trainGroup.shape[0]
    m = trainGroup.shape[1]

    a=[1 for x in range(n)]      #增广矩阵
    trainGroup = np.column_stack((trainGroup, a))

    w=[1 for x in range(m+1)]     #初始化W

    b=0.1                                #初始化B
    count=0                            #计数值，当count>向量个数numSamples，即完成训练
    cycle=100                            #循环次数，当超过循环次数，结束

    while(1):
        for i in range(n):
            cycle-=1
            print(w,trainGroup[i])      #每运行一步就打印一次W与对应的X
            if cal(trainGroup[i],trainLabels[i]) <= 0:               
                count=0
                w=w-direction*b*trainGroup[i]        #W=W-BX
            else:
                count+=1               
        if count >=n:           #判断是否取到适合的W
            print('Acomplish！')
            break
        elif cycle<=0:                   #失败
            print('Failed!')
            break


def cal(row,trainLabel):            #方向函数，用来判断X与W的方向
    global w, b,direction

    if np.matmul(row,w)==0:
        if trainLabel>0:
            direction=-1
        else:
            direction=1
        return 0
    elif np.matmul(row,w)>0:
        judge = 1
        if judge != trainLabel:
            direction=1
            return 0
    else:
        judge = -1
        if judge != trainLabel:
            direction=-1
            return 0
    return 1

def plotBestFit():   # 画图

    global w
    weights = w

    n = group.shape[0]
    fig = plt.figure() # 生成一个图片框

    ax = fig.add_subplot(1,1,1) # 编号
    for i in range(n):
        if labels[i] == 1:
            ax.scatter(group[i,0], group[i,1], color='r') #  红色
        else:
            ax.scatter(group[i, 0], group[i, 1], color='b')  #  蓝色
    x= range(0, 2,)
    y = (-weights[2]-weights[0]*x)/weights[1]
    ax.plot(x, y)
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


createDataSet()
perceptronClassify(group,labels)
plotBestFit()