import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy
from models.BaseModel import BaseModel


class SimpleKerasNNModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        model = Sequential([
            Dense(hidden_dim, input_dim=input_dim, activation='relu'),
            Dense(output_dim, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=[Accuracy()])
        super().__init__(model)

    def train(self, X, y, epochs=100, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X) -> pd.DataFrame:
        predictions = self.model.predict(X)
        return pd.DataFrame(predictions)

    def predict_single(self, x) -> int:
        prediction = self.predict(x)
        return 0 if prediction.iloc[0, 0] > 0.5 else 1

    def evaluate(self, X, y):
        loss, accuracy = self.model.evaluate(X, y, verbose=1)
        return {'loss': loss, 'accuracy': accuracy}

    def predict_proba(self, x) -> pd.DataFrame:
        probabilities = self.model.predict(x)
        probabilities_df = pd.DataFrame(probabilities)
        probabilities_df[0] = 1 - probabilities_df[0]
        probabilities_df[1] = 1 - probabilities_df[0]
        return probabilities_df
