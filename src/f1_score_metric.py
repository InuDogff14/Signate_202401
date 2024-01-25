# f1_score_metric.py
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import f1_score
import numpy as np

class F1ScoreMacro(Metric):
    def __init__(self):
        self._name = "f1_score_macro"
        self._maximize = True

    def __call__(self, y_true, y_pred):
        # 予測確率から最も可能性の高いクラスを選択
        y_pred_class = np.argmax(y_pred, axis=1)
        # F1スコア（マクロ平均）を計算
        f1score = f1_score(y_true, y_pred_class,average='weighted')
        return f1score