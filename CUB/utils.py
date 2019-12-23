import numpy as np
from numpy import *
from sklearn.metrics import accuracy_score
from config import *
from model import *
from numpy.random import seed
seed(1)

def data_iterator(sess, train_data, selected_num, img, att, weights):
    x = train_data['img_fea']
    train_label = train_data['tr_label']
    attribute = train_data['all_pro']

    unique_train_label = np.unique(train_label)
    batch_att = attribute[unique_train_label]  # 40 x 85

    dist = weight_distribute(img, att, weights)
    batch_fea = np.zeros(shape=(len(unique_train_label), 2048))
    for i in range(len(unique_train_label)):
        temp = np.where(train_label == unique_train_label[i])
        index = temp[0]
        idxs = np.arange(0,len(index))
        np.random.shuffle(idxs)
        select_idx = idxs[0:selected_num]
        select_fea = x[index[select_idx]]
        select_att = batch_att[i].reshape((1, len(batch_att[i])))
        mean_fea_a = sess.run(dist, feed_dict={img: select_fea, att: select_att})
        mean_fea_b = mean(select_fea,0)
        mean_fea = 0.2*mean_fea_a + 0.8*mean_fea_b
        batch_fea[i] = mean_fea

    batch_fea = batch_fea.astype("float32")
    batch_lab = range(len(unique_train_label))
    return batch_att, batch_fea, batch_lab

def compute_accuracy(gen_img, test_data):
    img_fea = test_data['img_fea']
    te_label = test_data['te_label']
    te_id = test_data['te_id']

    test_id = np.squeeze(np.asarray(te_id))
    outpre = [0]*img_fea.shape[0]
    test_label = np.squeeze(np.asarray(te_label))
    test_label = test_label.astype("float32")
    for i in range(img_fea.shape[0]):
        outputLabel = kNNClassify(img_fea[i,:], gen_img, test_id, 1)
        outpre[i] = outputLabel
    outpre = np.array(outpre, dtype='int')
    unique_labels = np.unique(test_label)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(test_label == l)[0]
        acc += accuracy_score(test_label[idx], outpre[idx])
    acc = acc / unique_labels.shape[0]
    return acc

def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]
    diff = tile(newInput, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = sum(squaredDiff, axis=1)
    distance_euc = 0*squaredDist ** 0.5
    distance_cos = [0] * dataSet.shape[0]
    for i in range(dataSet.shape[0]):
        distance_cos[i] = cosine_distance(newInput, dataSet[i])

    distance = distance_euc + distance_cos
    sortedDistIndices = argsort(distance)
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex

def cosine_distance(v1,v2):
    # compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    v1_sq = np.inner(v1,v1)
    v2_sq = np.inner(v2,v2)
    dis = 1 - np.inner(v1,v2) / math.sqrt(v1_sq * v2_sq)
    return dis
