import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

train_loss = np.load('train_loss.npy')
test_loss = np.load('test_losses.npy')
print("test", test_loss[-1])
print("train", train_loss[-1])
F1_train = np.load('F1_train.npy')
F1_test = np.load('F1_test.npy')
y_true_test = np.load('y_true_test.npy')
y_true_train = np.load('y_true_train.npy')
y_pred_train_prob = np.load('y_pred_train_prob.npy')
y_pred_test_prob = np.load('y_pred_test_prob.npy')

# Train Test loss graph
plt.plot(train_loss, label='train loss', color='hotpink')
plt.plot(test_loss, label='test loss', color='blue')
plt.legend()
plt.xlabel('ephoc')
plt.ylabel('loss')
plt.title('loss as function of ephocs')
plt.show()

# ROC AUC curve

fpr_test, tpr_test, _ = metrics.roc_curve(y_true_test, y_pred_test_prob)
auc_test = metrics.roc_auc_score(y_true_test, y_pred_test_prob)

fpr_train, tpr_train, _ = metrics.roc_curve(y_true_train, y_pred_train_prob)
auc_train = metrics.roc_auc_score(y_true_train, y_pred_train_prob)

print("Auc - Train:", auc_train)
print("Auc - Test:", auc_test)

plt.plot(fpr_test, tpr_test, color='hotpink', linewidth=6, label='test')
plt.plot(fpr_train, tpr_train, color='blue', label='train')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.legend()
plt.title('Roc-Auc curve for test and train')
plt.show()

# F1 train and test
plt.plot(F1_train, label='f1 score-train', color='hotpink')
plt.plot(F1_test, label='f1 score test', color='blue')
plt.legend()
plt.xlabel('ephoc')
plt.ylabel('F1 score')
plt.title('F1 score of train & test as function of ephocs')
plt.show()
