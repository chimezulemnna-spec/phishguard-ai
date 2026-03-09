"""
Model evaluation utilities.
"""

import matplotlib
matplotlib.use('Agg')           # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """Print a full evaluation report and return predictions."""
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])}")
    return y_pred


def plot_roc(model, X_test, y_test, save_path=None):
    """Plot and optionally save the ROC curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"✅ ROC curve saved to {save_path}")
    else:
        plt.show()
    plt.close()
    return auc