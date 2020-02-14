import numpy as np


def parsefile(filename):
    # 打开文件
    fr = open(filename)
    # 读取文件所有内容
    arrayLines = fr.readlines()
    # print(type(arrayLines))
    # 获取行数
    numberOfLine = len(arrayLines)
    # 创建arrayLine*3的矩阵
    returnMatrix = np.zeros((numberOfLine, 3))
    # 创建标签列表
    classlabelvector = []
    # 行索引
    index = 0

    for line in arrayLines:
        # 删除空白字符
        line = line.strip()
        # 将字符串用‘\t’进行分割切片
        listfromline = line.split('\t')
        # 提取前三列存入返回的矩阵中
        returnMatrix[index, :] = listfromline[0: 3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listfromline[-1] == 'didntLike':
            classlabelvector.append(1)
        elif listfromline[-1] == 'smallDoses':
            classlabelvector.append(2)
        elif listfromline[-1] == 'largeDoses':
            classlabelvector.append(3)
        index += 1
    return returnMatrix, classlabelvector

if __name__ == '__main__':
    filename = 'test_set.txt'
    datingDataMat, datinglabels = parsefile(filename)
    print(datingDataMat)
    print(datinglabels)
