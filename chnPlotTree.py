import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold
import sys
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False
import pickle

# Define the figure size and arrow format, program 3_5
decisionNode = dict(boxstyle="sawtooth", fc = "0.8")
leafNode = dict(boxstyle="round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

# Draw comment with arrows
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy = parentPt, \
    xycoords = 'axes fraction', xytext = centerPt, \
    textcoords = 'axes fraction', \
    va = "center", ha = "center", bbox = nodeType, \
    arrowprops = arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon = False)
    plotNode('Decision Node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('Branch Node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

# Number of leafs of tree, program 3_6
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # Testing whether the data type in node is dict
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': \
                    {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': \
                    {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]

# Fill text message in father nodes, program 3_7
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

# Calculate the height and width
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, \
                plotTree.yOff)
    # Mark the child node text
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # Reduce the shift value of y axis
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__== 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), \
                cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

    
pic1={'含糖率': {'>0.29': '1', '<=0.29': {'密度': {'>0.42': {'触感': {'硬滑': {'脐部': {'凹陷': {'纹理': {'清晰': {'敲声': {'浊响': {'根蒂': {'稍蜷': '1', '蜷缩': '1'}}, '沉闷': '1'}}, '稍糊': '0', '模糊': '1'}}, '稍凹': '1', '平坦': '0'}}, '软粘': '1'}}, '<=0.42': '1'}}}}

pic2={'触感': {'硬滑': {'脐部': {'凹陷': {'纹理': {'清晰': {'敲声': {'浊响': {'根蒂': {'蜷缩': '1', '稍蜷': '1', '硬挺': '1'}}, '沉闷': '1', '清脆': '1'}}, '稍糊': '0', '模糊': '1'}}, '稍凹': '0', '平坦': '0'}}, '软粘': {'色泽': {'青绿': '1', '乌黑': '1', '浅白': '1'}}}}


for i in range(2):
    eval('createPlot(pic'+str(i+1)+')')