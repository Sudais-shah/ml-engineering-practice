import tensorflow as tf
from tensorflow.keras import layers, losses, metrics, optimizers, models

class MLNN:
    def __init__(self, learning_rate: float = 0.01, input_shape: tuple = (784,)):
        
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.model = self._build_model()
        self._compile_model()
        
    
    def _build_model(self):
        model = models.Sequential( name = "MLNN") 
        model.add(layers.Dense(64, activation='relu', input_shape = self.input_shape))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model
    
    def _compile_model(self):
        self.model.compile(
            optimizer = optimizers.Adam(self.learning_rate),
            loss = losses.BinaryCrossentropy(),
            metrics = [metrics.BinaryAccuracy()]
        )
      
    def train_model(self, x_train, y_train, epochs: int = 10, batch_size: int = 32):
        return self.model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size)


    def evaluate_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    
    def predict_model(self, x_test):
        return self.model.predict(x_test)
    

    def summary(self):
        return self.model.summary()