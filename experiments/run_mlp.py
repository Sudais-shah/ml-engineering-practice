import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification 
from deep_learning.multi_layer_NN import MLNN


def main():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
                         n_repeated=0, n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, 
                         class_sep=1.0, hypercube=True, random_state=None)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    mlp = MLNN(learning_rate=0.01, input_shape=(20,))
    mlp.train_model(X_train, y_train)
    mlp.evaluate_model(X_test, y_test)
    mlp.predict_model(X_test)
    mlp.summary(
        
    )

if __name__ == "__main__": 
    main()