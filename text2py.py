from numpy import *
import operator

def creatDataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A',"B",'B']
    return group, labels

group, labels = creatDataset()

def classify0(inX, dataSet, labels, k):
    # inX is the input vector that needs to be classified
    # dataset: given labeled dataset
    # labels: dataset labels
    # k: num of choosing the most adjacent points
    dataSetSize = dataSet.shape[0] #num points of dataset
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # get numoffeatures* numofdatapoints
    sqDiffMat = power(diffMat,2)
    sqdistance = sqDiffMat.sum(axis=1)
    distance = power(sqdistance,0.5)  # calculate distance
    sorteddistindex = distance.argsort()  # get the index of distance from small to big [2,5,1,7,6...]
    classcount = { }
    for i in range(k):
        tlabel = labels[sorteddistindex[i]]   # find the label(i) for the i's nearest vector to inX labels[0]='A' tlabel=A
        classcount[tlabel] = classcount.get(tlabel,0) + 1  # 0 is the default value. used when labelorder not exist
    '''
    classcount[A]=1
    labels[1]='b' tlabel=b and classcount[b] = 1 and if labels[2]='a' then classcount[A]=1+1=2 classcount:{'A':2,'B':1}
    '''
    #    let's sort the counted labels
    sortcount = sorted(classcount.items(),key=operator.itemgetter(1),reverse=True) #from big to small
    return(sortcount[0][0])

def file2matrix(dir):
    with open(dir) as fr:
     lines = fr.readlines()
     numoflines = len(lines)
     tempMat = zeros((numoflines,3))
     classLabels = []
     index = 0
     for i in lines:
         i = i.strip()  # remove the space in head and tail
         listformline = i.split('\t')  # divide by tab get the content of a line
         tempMat[index,:] = listformline[0:3]  # get the first 3 elements and put it in a row of Mat
         classLabels.append(int(listformline[-1]))  # the last element = label
         index += 1
     return tempMat, classLabels

# newvalue=(oldvalue-min)/(max-min)
def normalization(dataset):
    minVals = dataset.min(0)  # get the minimum number of a column
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normdataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normdataset = dataset  - tile(minVals,(m,1))
    normdataset = normdataset/tile(ranges,(m,1))
    return normdataset, ranges, minVals


'''
the matrix is 1000*3; minvals is 1*3; ranges is 1*3 so we need tile() to broadcast our vector
to the same shape of matrix
'''

def datingtest():
    ratio = 0.1
    datingmat, datinglabel = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = normalization(datingmat)
    m = normMat.shape[0]
    numTestVec = int(m*ratio)
    errorcount = 0
    for i in range(numTestVec):
        classifierResult = classify0(normMat[i,:], normMat[numTestVec:m,:],
                                      datinglabel[numTestVec:m],1)
        print("the classifier came back with:"+str(classifierResult)+"the real answer is"
              +str(datinglabel[i]))
        if(classifierResult != datinglabel[i]):
            errorcount+=1
    print("the total error rate is"+str(errorcount/float(numTestVec)))

#print(classify0([0,0], group, labels, 6))