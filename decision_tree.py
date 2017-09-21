import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import sys


data = np.genfromtxt("data/X_train.txt")

Xtr = data[:10000] # 10000 points (of 14 features)
Xva = data[10000:20000]

data2 = np.genfromtxt("data/Y_train.txt")
Ytr = data2[:10000]
Yva = data2[10000:20000]

def test_depth(d):
    ''' For each depth 0 to d, print out the training and validation
    error if the maxDepth is set to d '''
    min_val_err = sys.maxint
    depth_min_err = 0
    for i in range(d+1):
        learner = ml.dtree.treeClassify(Xtr, Ytr, maxDepth=i)
        print 'For depth ', i, ':'
        print 'training error is', learner.err(Xtr, Ytr)
        val_err = learner.err(Xva, Yva)
        if(val_err < min_val_err):
            min_val_err = val_err
            depth_min_err = i
        print 'validation error is', val_err
        print
    print 'minimum validation error of', min_val_err, 'occured at depth of', depth_min_err

def test_leaf(d):
    ''' For each i from 0 to d, test training and error of the validation error
    if the minLeaf is set to 2^i'''
    min_val_err = sys.maxint
    leaf_min_err = 0
    for i in range(1,d+1):
        min_leaf = 2**i
        learner = ml.dtree.treeClassify(Xtr, Ytr, minLeaf=i)
        print 'For minLeaf ', min_leaf, ':'
        print 'training error is', learner.err(Xtr, Ytr)
        val_err = learner.err(Xva, Yva)
        if(val_err < min_val_err):
            min_val_err = val_err
            leaf_min_err = min_leaf
        print 'validation error is', val_err
        print
    print 'minimum validation error of', min_val_err, 'occured at minLeaf of', leaf_min_err
