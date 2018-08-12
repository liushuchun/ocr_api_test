# coding=utf-8  
# created by xiaqunfeng

import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import itertools
eps=1e-9

ap = argparse.ArgumentParser(description='script for ocr detection test')
ap.add_argument('--groundtruth','-g',type=str, required=True,
                help='groundtruth from labelX. XXX.json')
ap.add_argument('--result','-r',type=str, required=True,
                help='test result from refinedet or rfcn. XXX.tsv')
args = ap.parse_args()
gt_file=args.groundtruth
res_file=args.result

def get_groundtruth(gt_file):
    labels={}
    with open(gt_file,'r') as f:
        for line in f:
            line=json.loads(line)
            name=line['url'].split('/')[-1]
            if len(line['label'][0]['data']) == 0:
                continue
            label=line['label'][0]['data'][0]['class']
            if label=='ocr':
                label=0
            elif label=='normal':
                label=1
            labels[name]=label
        return labels

def get_true_pred(res_file,labels_dict):
    y_true=[]
    y_pred=[]
    with open(res_file,'r') as f:
        for line in f:
            #json_img = json.loads(line.split('\t')[1])
            json_img = eval(line.split('\t')[1])
            if len(json_img) == 0:
                label_pred = 1
            else:
                label_pred = 0
            name_img = line.split('\t')[0]
            if name_img in labels_dict:
                label_true = labels_dict[name_img.split('.')[0]]
                y_true.append(label_true)
                y_pred.append(label_pred)
        return y_true,y_pred

def confusion_matrix(y_true,y_pred):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    conf_matrix=np.zeros((2,2),dtype=int)

    for i in range(2):
        for j in range(2):
            conf_matrix[i][j]=np.where(y_pred[np.where(y_true==i)]==j)[0].shape[0]
    return conf_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')
    print(cm)
    print('\n')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('rfcn_confusion_matrix.jpg')

# top1 error(accuracy)
def accuracy(cnf_matrix):
    return float(cnf_matrix[0][0]+cnf_matrix[1][1])/(np.sum(cnf_matrix)+eps)

# ocr recall
def ocr_recall(cnf_matrix):
    return float(cnf_matrix[0][0]) / (np.sum(cnf_matrix[0]) + eps)

# ocr precision
def ocr_precision(cnf_matrix):
    return float(cnf_matrix[0][0])/(np.sum(cnf_matrix,axis=0)[0]+eps)

# normal recall
def normal_recall(cnf_matrix):
    return float(cnf_matrix[1][1])/(np.sum(cnf_matrix,axis=1)[1]+eps)

# normal precision
def normal_precision(cnf_matrix):
    return float(cnf_matrix[1][1])/(np.sum(cnf_matrix,axis=0)[1]+eps)

def main():
    ground_truth=get_groundtruth(gt_file)
    y_true,y_pred=get_true_pred(res_file,ground_truth)
    conf_matrix=confusion_matrix(y_true,y_pred)
    acc=accuracy(conf_matrix)
    tr=ocr_recall(conf_matrix)
    tp=ocr_precision(conf_matrix)
    nr=normal_recall(conf_matrix)
    np=normal_precision(conf_matrix)

    print('confusion matrix:\n%s'%conf_matrix)
    print('accuracy:         %s'%acc)
    print('ocr_recall:      %s'%tr)
    print('ocr_precision:   %s'%tp)
    print('normal_recall:    %s'%nr)
    print('normal_precision: %s'%np)

    labels = ['ocr', 'normal']
    plot_confusion_matrix(conf_matrix, labels)

if __name__=="__main__":
    main()