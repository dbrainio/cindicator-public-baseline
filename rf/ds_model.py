import io
import os
import sys

import dill
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

"""
More detailed example of DBrain API. You can definitely use it as a boilerplate for your complex models.

This model builds features for each question as mean and median predictions of top and worst-performing users
and trains a random forest model on those features to predict the true outcome.
"""


class DataTransformer:
    def __init__(self, top_acc_cutoff=0.75, worst_acc_cutoff=0.35,
                 window_size=30, active_user_npreds=10):
        """This class computes per-user statistics and aggregates user's predictions into question features"""
        self.window_size = window_size
        self.top_acc_cutoff = top_acc_cutoff
        self.worst_acc_cutoff = worst_acc_cutoff
        self.active_user_npreds = active_user_npreds

        self.user_accs = None

    def select_window(self, df):
        """select activity for latest window_size days"""
        cur_time = df["user_answer_created_at"].max()
        window_start_date = cur_time - pd.Timedelta(days=self.window_size)
        return df[df["user_answer_created_at"] >= window_start_date]

    def select_active_users(self, df):
        """select activity of users which were active enough"""
        activity_counter = df["user_id"].value_counts()
        active = activity_counter[activity_counter > self.active_user_npreds].index
        return df[df["user_id"].isin(active)]

    def compute_user_accuracy(self, df):
        """computes accuracy score for each active user"""
        df = self.select_window(df)
        df = self.select_active_users(df)
        user_accs = {}
        for user_id, group in df.groupby("user_id"):
            preds = group["user_answer"]
            targets = group["question_answer"]
            acc = accuracy_score(targets, preds > 0.5)
            user_accs[user_id] = acc
        self.user_accs = user_accs

    def transform(self, df, include_target=True):
        """builds features: mean and median predictions for top, worst and regular users"""

        top_users = [user_id for user_id, acc in self.user_accs.items() if acc >= self.top_acc_cutoff]
        worst_users = [user_id for user_id, acc in self.user_accs.items() if acc <= self.worst_acc_cutoff]

        features = []
        for question_id, group in df.groupby("question_id"):

            user_ids = group["user_id"]
            preds = group["user_answer"]

            top_preds = []
            worst_preds = []
            regular_preds = []

            for user_id, pred in zip(user_ids, preds):
                if user_id in top_users:
                    top_preds.append(pred)
                elif user_id in worst_users:
                    worst_preds.append(pred)
                else:
                    regular_preds.append(pred)

            row = {
                "question_id": question_id,
                "avg_top_pred": np.mean(top_preds),
                "median_top_pred": np.median(top_preds),
                "avg_worst_pred": np.mean(worst_preds),
                "median_worst_pred": np.median(worst_preds),
                "avg_regular_preds": np.mean(regular_preds),
                "median_regular_preds": np.median(regular_preds)
            }

            if include_target:
                row["question_answer"] = np.mean(group["question_answer"])
            features.append(row)
        self.featnames = [k for k in features[0].keys() if not k in ("question_id", "question_answer")]
        features = pd.DataFrame(features)
        features = features.fillna(value=0)
        return features


class DSModel:
    def __init__(self, assets_dir: str, dump_dir: str):
        """
        We declare persistent variables like model and transformer instances, auxiliary paths and so on.
        :param assets_dir: A path to directory where your assets are loaded
        :param dump_dir: A path to persistent directory which should be used
                                for saving data and models at the end of training.
                             Everything that you save here is guaranteed to be available at subsequent stages.
        """

        self.assets_dir = assets_dir
        self.dump_dir = dump_dir

        self.transformer = DataTransformer()
        self.model = None
        self.scaler = None

        self.id_column = "question_id"
        self.target_column = "question_answer"

        self.df = None

    def train(self, data_dir: str) -> None:
        """
        A method that is called first. Pre-trains your model on the training data.
        At the end of .train() call .dump is called to save your model.

        data_dir contains multiple csv files, each one for questions finished at the same time.

        Here we implement a simple random forest model.

        :param data_dir: A path to directory where data and markup is stored. For data format please refer to
        :return:
        """

        # Join all separate files into a single data frame
        train_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(".csv")]
        df = pd.concat([pd.read_csv(fname) for fname in train_files])
        df["finished_question_time"] = pd.to_datetime(df["finished_question_time"])
        df["user_answer_created_at"] = pd.to_datetime(df["user_answer_created_at"])
        self.df = df

        self.transformer.compute_user_accuracy(self.df)
        feats = self.transformer.transform(self.df, include_target=True)
        X = feats[self.transformer.featnames]
        y = feats["question_answer"]
        self.scaler = StandardScaler().fit(X)
        self.model = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=4, n_estimators=200).fit(X, y)

    def dump(self) -> None:
        """
        Called at the end of .train() automatically. Save everything you need for prediction here!

        We save pickles of all that is needed to build a model from static dumps at test time.
        Right now we fit and tune the dataloader, scaler and the model, so we dump those objects.
        :return:
        """
        joblib.dump(self.model, self.dump_dir + "/model.jbl")
        joblib.dump(self.df, self.dump_dir + "/df.jbl")
        joblib.dump(self.scaler, self.dump_dir + "/scaler.jbl")
        joblib.dump(self.transformer, self.dump_dir + "/transformer.jbl")

    def load(self) -> None:
        """
        Called when your model is instantiated before scoring for prediction. Prepares your DSModel for scoring.

        We just load all objects fitted during .train() from according pickles.
        :return:
        """
        self.df = joblib.load(self.dump_dir + "/df.jbl")
        self.model = joblib.load(self.dump_dir + "/model.jbl")
        self.scaler = joblib.load(self.dump_dir + "/scaler.jbl")
        self.transformer = joblib.load(self.dump_dir + "/transformer.jbl")

    def predict(self, items: [bytes]) -> [{str: {str: object}}]:
        """
        This method is called to retrieve a prediction of your model for a batch of data. Data format depends on a contest

        :param items: In this contest items is a list of single item, where item is a bytes-encoded csv,
                                                                        similar to those in train
        :return: [{target_name: {object_id: prediction}}]
        """

        # read bytes into dataframe
        stream = io.BytesIO(items[0])
        df = pd.read_csv(stream)

        df["finished_question_time"] = pd.to_datetime(df["finished_question_time"])
        df["user_answer_created_at"] = pd.to_datetime(df["user_answer_created_at"])

        feats = self.transformer.transform(df, include_target=False)
        X = feats[self.transformer.featnames].as_matrix()
        X = self.scaler.transform(X)
        preds = self.model.predict_proba(X)[:, 1]

        results = {}
        for qid, p in zip(feats["question_id"], preds):

            results[int(qid)] = p

        return [{'question_answer': results}]

    def update(self, items: [(bytes, bytes)]) -> None:
        """
        Is called after .predict() to update your model.
        Input is a pair of previous batch items and according targets (X, y).
        :param items: Bytes-encoded pair of (X, y), where X is the same data that was passed to .predict() call
        :return:
        """
        x_df = pd.concat([pd.read_csv(io.BytesIO(bytestream[0]), index_col=0) for bytestream in items])
        target_df = pd.concat([pd.read_csv(io.BytesIO(bytestream[1]), index_col=0) for bytestream in items])
        update_df = pd.merge(x_df, target_df, right_on="id", left_on="question_id")

        update_df["finished_question_time"] = pd.to_datetime(update_df["finished_question_time"])
        update_df["user_answer_created_at"] = pd.to_datetime(update_df["user_answer_created_at"])

        self.df = pd.concat([self.df, update_df])

        self.transformer.compute_user_accuracy(self.df)
        feats = self.transformer.transform(self.df, include_target=True)
        X = feats[self.transformer.featnames]
        y = feats["question_answer"]
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.model = self.model.fit(X, y)


if __name__ == '__main__':
    """
        A basic test of your model. Trains the model on all files from data_dir and predicts for a single file in data_dir.
        For thorough testing please refer to ModelTraining.ipynb
    """
    asset_dir = 'asset_dir'
    dump_dir = 'dump_dir'
    data_dir = 'data_dir'

    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        model = DSModel(asset_dir, dump_dir)
        print("train")
        model.train(data_dir)
        model.dump()
    elif len(sys.argv) == 2 and sys.argv[1] == 'test':
        # load just a single file for test
        filename = os.path.join(str(data_dir), '00000.csv')
        with open(filename, 'rb') as file:
            bytes_ = file.read()
        input_data = [bytes_]

        model = DSModel(asset_dir, dump_dir)
        model.load()
        results = model.predict(input_data)
        print(results)
    else:
        raise Exception
