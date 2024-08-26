import json

import pandas as pd
from sklearn.neighbors import KernelDensity

from tqdm import tqdm
from algorithms.mtad_gat.eval_methods import *
from algorithms.mtad_gat.utils import *


def get_cdf_threshold_50_percent(train_anomaly, threshold):
    kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(train_anomaly.reshape(-1, 1))
    x = np.linspace(0, 10, 1000).reshape(-1, 1)  # 0, 0.01, 0.02, 0.03, ...
    log_pdf = kde.score_samples(x)
    pdf = np.exp(log_pdf)

    # threshold 보다 큰 구간의 분포
    anomaly_pdf = pdf[(x > threshold).reshape(-1)]
    anomaly_cdf = 0.95 * np.cumsum(anomaly_pdf) / np.sum(anomaly_pdf)

    # threshold 보다 작은 구간의 분포
    normal_pdf = pdf[(x <= threshold).reshape(-1)]
    normal_cdf = np.array([0] * len(normal_pdf))

    total_cdf = np.r_[normal_cdf, anomaly_cdf]

    # CDF 50퍼센트 지점
    threshold_50_percent = (1000 - (total_cdf > 0.5).sum()) * 0.01

    return total_cdf, threshold_50_percent


class Predictor:
    """MTAD-GAT predictor class.

    :param model: MTAD-GAT model (pre-trained) used to forecast and reconstruct
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param pred_args: params for thresholding and predicting anomalies

    """

    def __init__(self, model, event, window_size, n_features, pred_args, summary_file_name="summary.txt"):
        self.model = model
        self.event = event
        self.window_size = window_size
        self.n_features = n_features
        self.columns = pred_args["columns"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.batch_size = 256
        self.use_cuda = pred_args["use_cuda"]
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name

    def get_score(self, values):
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :return np array of anomaly scores + dataframe with prediction for each channel and global anomalies
        """

        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims, predict=True)  # predict True: 전체 데이터 사용
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        preds = []
        recons = []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)

                # y_hat, _, _ = self.model(x)

                # Shifting input to include the observed value (y) when doing the reconstruction
                recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                _, window_recon = self.model(recon_x)

                # preds.append(y_hat.detach().cpu().numpy())
                # preds.append(y_hat[:, 0, :].detach().cpu().numpy())
                # Extract last reconstruction only
                recons.append(window_recon[:, -1, :].detach().cpu().numpy())

        # preds = np.concatenate(preds, axis=0)
        recons = np.concatenate(recons, axis=0)
        actual = values.detach().cpu().numpy()[self.window_size:]

        if self.target_dims is not None:
            actual = actual[:, self.target_dims]

        anomaly_scores = np.zeros_like(actual)
        df = pd.DataFrame()
        for i in range(recons.shape[1]):
            # a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2) + self.gamma * np.sqrt(
            #     (recons[:, i] - actual[:, i]) ** 2)
            a_score = np.sqrt((recons[:, i] - actual[:, i]) ** 2)

            if self.scale_scores:
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                a_score = (a_score - median) / (1 + iqr)

            anomaly_scores[:, i] = a_score
        # 각 지표 anomaly score의 평균
        df['ETC'] = np.mean(anomaly_scores, axis=1)

        event_feature_num_dict = dict()
        for event_id, event_features in self.event.items():
            event_feature_num_dict[event_id] = list()
            for feature_n in event_features:
                event_feature_num_dict[event_id].append(self.columns.index(feature_n))

        for key, values in event_feature_num_dict.items():
            event_scores = anomaly_scores[:, values].sum(axis=1)
            df[f'{key}'] = event_scores / len(values)

        return df, actual, recons

    def predict_anomalies(self, train):
        """ Predicts anomalies

        :param train: 2D array of train multivariate time series data
        :param test: 2D array of test multivariate time series data
        :param true_anomalies: true anomalies of test set, None if not available
        :param save_scores: Whether to save anomaly scores of train and test
        :param load_scores: Whether to load anomaly scores instead of calculating them
        :param save_output: Whether to save output dataframe
        :param scale_scores: Whether to feature-wise scale anomaly scores
        """

        train_pred_df, actual, recons = self.get_score(train)

        threshold = dict()
        threshold_50_percent = dict()
        anomaly_cdf = dict()
        train_glb_anomaly_score = train_pred_df["ETC"].values
        glb_p_eval = pot_eval(train_glb_anomaly_score, train_glb_anomaly_score, None,
                          q=self.q, level=self.level, dynamic=self.dynamic_pot)
        for k, v in glb_p_eval.items():
            if not type(glb_p_eval[k]) == list:
                glb_p_eval[k] = float(v)
        threshold["ETC"] = glb_p_eval["threshold"]
        anomaly_cdf["ETC"], threshold_50_percent["ETC"] = get_cdf_threshold_50_percent(train_glb_anomaly_score, threshold["ETC"])

        for event_key in train_pred_df.columns:
            train_anomaly_scores = train_pred_df[event_key].values

            if self.use_mov_av:
                smoothing_window = int(self.batch_size * self.window_size * 0.05)
                train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(
                    span=smoothing_window).mean().values.flatten()

            # e_eval = epsilon_eval(train_anomaly_scores, train_anomaly_scores, None, reg_level=self.reg_level)
            p_eval = pot_eval(train_anomaly_scores, train_anomaly_scores, None,
                              q=self.q, level=self.level, dynamic=self.dynamic_pot)

            for k, v in p_eval.items():
                if not type(p_eval[k]) == list:
                    p_eval[k] = float(v)

            threshold[f"{event_key}"] = max(p_eval["threshold"], glb_p_eval["threshold"])
            anomaly_cdf[f"{event_key}"], threshold_50_percent[f"{event_key}"] = get_cdf_threshold_50_percent(train_anomaly_scores, threshold[f"{event_key}"])

        return threshold_50_percent, anomaly_cdf, actual, recons