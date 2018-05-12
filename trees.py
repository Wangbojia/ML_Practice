from math import log
import operator

def calEntrophy(dataset):
    numEntries = len(dataset)
    label_counts = {}
    for feature_vec in dataset:
        current_label = feature_vec[-1]  # only focus on the last feature this time
        if current_label not in label_counts.keys():  # dic.keys(): return keys in dic in list
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    Entrophy = 0
    for key in label_counts:
        prob = float(label_counts[key])/numEntries
        Entrophy -= prob*log(prob,2)
    return Entrophy

def creatdataset():
    dataset = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [1,0,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataset, labels


def splitdata(dataset,axis,value):  # axis: which column should be used; value: our expected value for this column
    retdataset = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducedfeatvec = featvec[:axis]  # the axis'th matches,extract the first axises features
            reducedfeatvec = reducedfeatvec + featvec[axis+1:]  # combine with the last; in other word we delete the axis'th feature
            retdataset.append(reducedfeatvec)  # regenerate the list to avoid editing the object directly
        return retdataset


def choose_best_feature(dataset):
    num_features = len(dataset[0])-1  # the last one is label
    base_entropy = calEntrophy(dataset)
    best_entro_gain = 0
    best_feature = -1
    feat_list = []
    for i in range(num_features):
        for example in dataset:
            feat_list.append(example[i])
        unique_val = set(feat_list)  # use set function to remove features appear more than once; decrease the calc
        new_entropy = 0
        for value in unique_val:
            subdataset = splitdata(dataset, i, value)
            prob = len(subdataset)/float(len(dataset))  # the subdata is choosing by our selected features
            new_entropy += prob*calEntrophy(subdataset)
        info_gain = base_entropy - new_entropy  # the larger the entro, the orderless the data;
        if info_gain>best_entro_gain:
            best_entro_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(classlist):
    # used to return the sorted list by the occurrence of elements
    class_count = {}
    for vote in classlist:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote]+=1
    sorted_classcount = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_classcount[0][0]  # similar method to text2py

def create_tree(dataset,labels):
    class_list = []
    for example in dataset:
        class_list.append(example[-1])
    if class_list.count(class_list[0]) == len(class_list):  # see if the list is the repeat of the first element
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_count(class_list)