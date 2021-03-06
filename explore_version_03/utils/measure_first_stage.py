from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import csv
from sklearn.metrics  import roc_curve,auc

class MeasureR(object):
    def __init__(self, fdir, loss, acc):
      self.fdir = fdir
      self.file_path = os.path.join(self.fdir, 'result_detail_resnet152_valid_cv5.csv')
      self.wfile_path = os.path.join(self.fdir, 'measure_detail_resnet152_valid_cv5.csv')
      
      self.acc = acc
      self.loss = loss
      print (self.fdir)
      print (self.file_path)
      print (self.acc, self.loss)
      
    def output(self):
      with open(self.file_path, 'r') as f:
        csv_reader = csv.reader(f)
        p0 = []
        p1 = []
        l0 = []
        l1 = []
        
        target_s = np.zeros(2).astype(float)
        predict_s = np.zeros(2).astype(float)
        tp_s = np.zeros(2)
        for row in csv_reader:
          #print (row)
          pv = np.array(row[:2])
          rv = np.array(row[2:])
          #print(row[:2])
          #print(row[2:])
          #print(int(bool(int(float(row[2]))) or bool(int(float(row[3]))) or bool(int(float(row[4])))))
          #print (pv, rv)
          #if int(bool(int(float(row[2]))) or bool(int(float(row[3]))) or bool(int(float(row[4])))) == 1:
          #  t_id = 1
          #else:
          #  t_id = 0
          p_id = np.argmax(pv)
          t_id = np.argmax(rv)
          #print(p_id, t_id)
          target_s[t_id] += 1.
          predict_s[p_id] += 1.
          #print(target_s, predict_s)
          if t_id == p_id:
            tp_s[t_id] += 1.
          p0.append(float(row[0]))
          p1.append(float(row[1]))
          # if int(float(row[2])) == 1:
          #   l1.append(int(float(row[2])))
          # elif int(float(row[3])) == 1:
          #   l1.append(int(float(row[3])))
          # elif int(float(row[4])) == 1:
          #   l1.append(int(float(row[4])))
          # else:
          #   l1.append(0)
          # l0.append(int(float(row[5])))
          l0.append(float(row[2]))
          l1.append(float(row[3]))

        p0 = np.array(p0)  
        p1 = np.array(p1)
        l0 = np.array(l0)
        l1 = np.array(l1)

        precision = tp_s/predict_s
        recall = tp_s/target_s 
        with open(self.wfile_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Type','normal','non_normal'])            
            csv_writer.writerow(['Precision']+list(precision))
            csv_writer.writerow(['Recall']+list(recall))
        fpr_0,tpr_0,threshold_0=roc_curve(l0,p0)
        roc_auc_0=auc(fpr_0,tpr_0)
        fpr_1,tpr_1,threshold_1=roc_curve(l1,p1)
        roc_auc_1=auc(fpr_1,tpr_1)
        plt.figure(figsize=(10,10))
        plt.plot(fpr_0, tpr_0, color='red',
        lw=2, label='normal (AUC = %0.4f)' % roc_auc_0) ###??????????????????????????????????????????????????????
        plt.plot(fpr_1, tpr_1, color='black',
        lw=2, label='non_normal (AUC = %0.4f)' % roc_auc_1) ###??????????????????????????????????????????????????????

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.text(.6, .5, 'Test Loss: %f, Acc: %f'%(self.loss, self.acc))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        foo_fig = plt.gcf()
        saved_path = os.path.join(self.fdir, 'results.png')
        if os.path.exists(saved_path):
          os.remove(saved_path)
        foo_fig.savefig(saved_path)
        plt.show()

if __name__ == '__main__':
    dir1 = './explore_version_03/results/resnet152_20200407_multiclass_cv5'
#    dir2 = './results/resnet18_20200322_weight_1vs100_1_100_woRegularization'
#    dir3 = './results/resnet18_20200322_weight_1vs6_15_100_woRegularization_augumentation'
#    dir4 = './results/resnet18_20200322_weight_1vs6_15_100_woRegularization_stadndard'
    mr1 = MeasureR(dir1, 0.1, 0.1)
    mr1.output()
#    mr2 = MeasureR(dir2, 0.1, 0.1)
#    mr2.output()
#    mr3 = MeasureR(dir3, 0.1, 0.1)
#    mr3.output()
#    mr4 = MeasureR(dir4, 0.1, 0.1)
#    mr4.output()

#    mr = MeasureR(dir1, 0.1, 0.1)