import numpy as np
import operator


def creatdataset():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    lables = ['爱情片', '爱情片', '动作片', '动作片']
    return group, lables


'''
    how to calculate the Euclidean distance(欧式距离) of two matrices :
    step1 :
        do matrix subtraction
    step2 :
        do matrix square
    step3 :
        do matrix column addition (matrix.sun( axis = 1 ))
    step4 :
        do matrix sqrt
'''

'''
    function description : calculate the Euclidean distance of two matrices
    parameters : 
        test_set : data of test set
        training_set : data of training set
        labels : classified labels
        k : select k points with minimum distance of 
    returns : 
        
    modify :
        2019/9/24
'''
def classify(test_set, training_set, lables, k):
    # get the numbers of rows in training set
    datasetsize = training_set.shape[0]
    # print(datasetsize)
    # add the dimension of the test set and subtract it from training set
    '''
        the effect of np.tile():
        // eg. np.tile(test_set , (n))
        if n == 2
        the test_set will be doubled to the right
        such as:
        test_set = np.array(
                   [[1,2,3],
                    [4,5,6],
                    [7,8,9]]
                   )
        the test_set will become :
        [[1 2 3 1 2 3]
         [4 5 6 4 5 6]
         [7 8 9 7 8 9]]
   '''
    diffMat = np.tile(test_set, (datasetsize, 1)) - training_set
    # print(diffMat)
    # do matrix square
    sqdiffMat = diffMat**2
    # print(sqdiffMat)
    # do matrix column addition (matrix.sun( axis = 1 ))
    sqDistances = sqdiffMat.sum(axis=1)
    # print(type(sqDistances))
    # print(sqDistances)
    # do matrix sqrt
    distances = sqDistances ** 0.5
    # print(distances)
    # return the index of sorted distances matrix
    sortedDistIndices = distances.argsort()
    # print(sortedDistIndices[0])
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        # print(voteIlabel)
        # print(type(voteIlabel))
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # print(classCount[voteIlabel])
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    # print(classCount)
    # print(classCount.items())
    # print(type(classCount.items()))
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    # print(sortedClassCount)
    # return the max count of key in sorted list
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = creatdataset()
    test = [101, 20]
    test_class = classify(test, group, labels, 3)
    print(test_class)