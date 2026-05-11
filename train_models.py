import numpy as np
import json
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)
target_names = list(data.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=10000, random_state=42)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
y_prob_lr = lr.predict_proba(X_test_s)[:,1].tolist()

nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
nn.fit(X_train_s, y_train)
y_pred_nn = nn.predict(X_test_s)
y_prob_nn = nn.predict_proba(X_test_s)[:,1].tolist()

lr_coef = lr.coef_[0].tolist()
lr_intercept = float(lr.intercept_[0])
scaler_mean = scaler.mean_.tolist()
scaler_std = scaler.scale_.tolist()

nn_weights = []
for w, b in zip(nn.coefs_, nn.intercepts_):
    nn_weights.append({'weights': w.tolist(), 'biases': b.tolist()})

def metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        'confusion_matrix': cm,
        'accuracy': round(accuracy_score(y_true, y_pred)*100, 2),
        'precision': round(precision_score(y_true, y_pred)*100, 2),
        'recall': round(recall_score(y_true, y_pred)*100, 2),
        'f1': round(f1_score(y_true, y_pred)*100, 2)
    }

model_data = {
    'feature_names': feature_names,
    'target_names': target_names,
    'scaler': {'mean': scaler_mean, 'std': scaler_std},
    'logistic_regression': {
        'coef': lr_coef,
        'intercept': lr_intercept,
        'test_metrics': metrics(y_test, y_pred_lr),
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'y_pred': y_pred_lr.tolist(),
        'y_prob': y_prob_lr
    },
    'neural_network': {
        'layers': nn_weights,
        'n_layers': len(nn.coefs_),
        'test_metrics': metrics(y_test, y_pred_nn),
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'y_pred': y_pred_nn.tolist(),
        'y_prob': y_prob_nn
    }
}

with open('/home/claude/model_data.json', 'w') as f:
    json.dump(model_data, f)

print("LR Accuracy:", round(accuracy_score(y_test, y_pred_lr)*100,2), "%")
print("NN Accuracy:", round(accuracy_score(y_test, y_pred_nn)*100,2), "%")
print("Features:", len(feature_names))
print("Test samples:", len(y_test))
print("Done!")
