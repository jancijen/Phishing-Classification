from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

sns.set_style('white')

def evaluate_classifier(y_true, y_pred, y_pred_prob, metrics, sample_weights=None):
    # Metrics
    if metrics:
        print('Metric values:\n')
        
        for metric_name, metric_fn in metrics:
            print('{}: {:.3f}'.format(metric_name, metric_fn(y_true, y_pred, sample_weight=sample_weights)))
        
        print('\n')
        
    # Confusion matrix
    cm_labels = [True, False]
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels,sample_weight=sample_weights)
    cm_df = pd.DataFrame(cm/np.sum(cm, axis=0), index=cm_labels, columns=cm_labels)
    # Plot confusion matrix
    ax = sns.heatmap(round(cm_df,2), annot=True, fmt=',')
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix')
    plt.show()
