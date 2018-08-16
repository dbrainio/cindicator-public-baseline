import io

import pandas as pd


"""
Let's get familiar with the basics of Dbrain API!
A minimalistic submission code that predicts 0 for all questions. 
"""


class DSModel:
    def __init__(self, assets_dir: str, dump_dir: str):
        """
            :param assets_dir: A path to directory where your assets are loaded
            :param dump_dir: A path to persistent directory which should be used
                                for saving data and models at the end of training.
                             Everything that you save here is guaranteed to be available at subsequent stages.
        """
        self.id_column = "question_id"

    def train(self, data_dir: str) -> None:
        """
        A method that is called first. Pre-trains your model on the training data.
        At the end of .train() call .dump is called to save your model.
        btw, you can write to dump_dir anytime if you wish

        :param data_dir: A path to directory where data and markup is stored. For data format please refer to
        :return:

        """
        pass

    def dump(self) -> None:
        """
        Called at the end of .train() automatically. Save everything you need for prediction here!
        :return:
        """
        pass

    def load(self) -> None:
        """
        Called when your model is instantiated before scoring for prediction. Prepares your DSModel for scoring.
        :return:
        """
        pass

    def predict(self, items: [bytes]) -> [{str: {str: object}}]:
        """
        This method is called to retrieve a prediction of your model for a batch of data
        :param items: Bytes-encoded batch of items to predict on. Data format depends on a contest
        :return:
        """
        for x in items:
            stream = io.BytesIO(x)
            x_df = pd.read_csv(stream)
            question_ids = x_df[self.id_column]
            prediction = [{"question_answer": {q_id: 0.5 for q_id in question_ids}}]

        return prediction

    def update(self, items: [(bytes, bytes)]) -> None:
        """
        Is called after .predict() to update your model.
        Input is a pair of previous batch items and according targets (X, y).
        :param items: Bytes-encoded pair of (X, y), where X is the same data that was passed to .predict() call
        :return:
        """
        pass
