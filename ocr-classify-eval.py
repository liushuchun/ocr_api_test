# -*- coding: utf-8 -*-
# 分类评估

import os
import argparse
import utils

def parse():
    args = argparse.ArgumentParser('ocr class evaluation tool')
    #args.add_argument('mode',type = str,choices = ['log','infer'],help = '直接根据日志评估或先推理后评估')
    args.add_argument('--gt',type = str,required = True, help = 'LabelX标注过的json文件')
    args.add_argument('--log',type = str,required = True, help = '日志文件，每行一个结果（name\tclass）')
    args.add_argument('--label',type = str,required = True,help = '标签映射文件（index\tclassname）')
    args.add_argument('--verbose',action = 'store_true',default = False,help = '指定时绘制混淆矩阵')
    return args.parse_args()

def _read_label_file(label_file):
    """
    return class_label_dict,label_class_dict
    Args:
        label_file  --label map file
        syntax:
            index\tclassname
            ...
    """
    class_label_dict = dict()
    label_class_dict = dict()
    with open(label_file,'r') as f:
        for line in f:
            index,classname = line.strip().split('\t')
            class_label_dict[classname] = int(index)
            label_class_dict[int(index)] = classname
    return class_label_dict,label_class_dict

def evaluation():
    args = parse()
    class_label_dict,label_class_dict = _read_label_file(args.label)
    classes = list()
    num_classes = len(class_label_dict)
    for i in range(num_classes):
        classes.append(label_class_dict[i])
    gt_dict = utils.read_json(args.gt,class_label_dict)
    pred_dict = utils.read_log(args.log,class_label_dict)
    y_true,y_pred = utils.gen_yture_ypred(gt_dict,pred_dict)
    metrics = utils.Metrics(y_true,y_pred,classes)
    conf_matrix = metrics.confusion_matrix
    metrics_list = metrics.metrics_list()

    # print result
    print("class evaluation")
    print("~"*50)
    print("accuracy:      %.6f"%(metrics_list[-1]))
    print("~"*50)
    for i in range(num_classes):
        print("%s_recall:     %.6f"%(label_class_dict[i],metrics_list[i*2]))
        print("%s_precision:  %.6f"%(label_class_dict[i],metrics_list[i*2+1]))
        print("~"*50)
    print("Confusion Matrix")
    print(conf_matrix)
    if args.verbose:
        metrics.plot_confusion_matrix()
    print("Done.")

if __name__ == '__main__':
    evaluation()