
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

import re

def preprocess_text(text):
    """
    Preprocess text for transformer models
    - Remove HTML tags
    - Remove URLs
    - Remove mentions (@username)
    - Remove hashtags (#hashtag -> hashtag or removed)
    - Normalize whitespace
    """

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags (keep the word, remove #)
    text = re.sub(r'#(\w+)', r'\1', text)
    # remove hashtag with word
    # text = re.sub(r'#\w+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text





def plot_confusion_matrix(y_true, y_pred, title, target_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matrice de Confusion - {title}')
    plt.ylabel('Vrai Label')
    plt.xlabel('Label Prédit')
    plt.show()
    


def plot_multiclass_roc(y_test_bin, y_score, title, n_classes, target_names):
    # Calcul des courbes ROC pour chaque classe (One-vs-Rest)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC {target_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title(f'Courbes ROC Multi-classes - {title}')
    plt.legend(loc="lower right")
    plt.show()



def plot_learning_curve_graph(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score Entraînement")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score Validation (CV)")
    plt.title(f'Courbe d\'Apprentissage - {title}')
    plt.xlabel("Taille du jeu d'entraînement")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()