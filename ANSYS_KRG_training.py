import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from smt.surrogate_models import KRG
from sklearn.metrics import mean_squared_error, r2_score

class KrigingModelTrainer:
    def __init__(self, db_path='data_ansys.db'):
        self.db_path = db_path
        self.work_dir = self._get_work_dir()
        self.krg_dir = os.path.join(self.work_dir, "krg_models")
        os.makedirs(self.krg_dir, exist_ok=True)

    def _get_work_dir(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT path FROM path WHERE id=2")
            work_dir = cursor.fetchone()[0]
            conn.close()
            if not os.path.isdir(work_dir):
                raise FileNotFoundError(f"工作路径不存在: {work_dir}")
            return work_dir
        except Exception as e:
            raise RuntimeError(f"获取工作路径失败: {e}")

    def _load_data(self):
        try:
            train_file = os.path.join(self.work_dir, 'krg_train_data.csv')
            test_file = os.path.join(self.work_dir, 'krg_val_data.csv')

            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            return train_df, test_df
        except Exception as e:
            raise RuntimeError(f"读取训练/验证数据失败: {e}")

    def train_models(self, target_dict=None):
        if target_dict is None:
            target_dict = {'mass': 'Mass', 'stress': 'Stress'}

        train_df, test_df = self._load_data()
        input_columns = [col for col in train_df.columns if col not in target_dict.values()]
        X_train = train_df[input_columns].values
        X_test = test_df[input_columns].values

        results = {}  # 存储每个模型评估文本

        for name, col in target_dict.items():
            y_train = train_df[col].values.reshape(-1, 1)
            y_test = test_df[col].values.reshape(-1, 1)

            model = KRG(print_global=False)
            model.set_training_values(X_train, y_train)
            model.train()

            # 保存模型
            model_path = os.path.join(self.krg_dir, f"{name}_krg.pkl")
            joblib.dump(model, model_path)

            # 模型评估
            y_pred = model.predict_values(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            result_text = f"{name.upper()} 模型评估：\nRMSE: {rmse:.4f}\nR² Score: {r2:.4f}"
            results[name] = result_text
            print(result_text)

            self._plot_prediction(y_test, y_pred, name)

        return results

    def _plot_prediction(self, y_true, y_pred, name):
        plt.figure()
        plt.scatter(y_true, y_pred, c='blue', label='prediction vs real')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='ideal prediction')
        plt.xlabel("real")
        plt.ylabel("prediction")
        plt.title(f"{name.upper()} Kriging prediction")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()
if __name__ == '__main__':
    trainer = KrigingModelTrainer()
    trainer.train_models()