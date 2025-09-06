import os
import sqlite3
import numpy as np
import joblib
import matplotlib.pyplot as plt


class KrigingPSOOptimizer:
    def __init__(self, db_path='data_ansys.db'):
        self.db_path = db_path
        self.work_dir = self._get_work_dir()
        self.krg_dir = os.path.join(self.work_dir, "krg_models")
        os.makedirs(self.krg_dir, exist_ok=True)

        self.mass_model = None
        self.stress_model = None
        self.bounds = []

        self._load_models()
        self._load_variable_bounds()

    def _get_work_dir(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM path WHERE id=2")
        work_dir = cursor.fetchone()[0]
        conn.close()
        return work_dir

    def _load_models(self):
        self.mass_model = joblib.load(os.path.join(self.krg_dir, "mass_krg.pkl"))
        self.stress_model = joblib.load(os.path.join(self.krg_dir, "stress_krg.pkl"))

    def _load_variable_bounds(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT lower_bound, upper_bound FROM variables")
        rows = cursor.fetchall()
        conn.close()
        self.bounds = [(row[0], row[1]) for row in rows]

    def _objective(self, x):
        mass = self.mass_model.predict_values(x.reshape(1, -1))[0][0]
        stress = self.stress_model.predict_values(x.reshape(1, -1))[0][0]
        penalty = 1e6 * max(0, stress - 312) ** 2
        return mass + penalty

    def run_pso(self, n_particles=30, n_iterations=100, w=0.5, c1=1.5, c2=1.5):
        dim = len(self.bounds)
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])

        X = np.random.uniform(lb, ub, (n_particles, dim))  # 粒子位置
        V = np.zeros((n_particles, dim))                   # 速度
        pbest = X.copy()
        pbest_val = np.array([self._objective(x) for x in X])

        gbest_idx = np.argmin(pbest_val)
        gbest = pbest[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]
        history = [gbest_val]

        for t in range(n_iterations):
            r1, r2 = np.random.rand(n_particles, dim), np.random.rand(n_particles, dim)
            V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
            X = np.clip(X + V, lb, ub)

            current_val = np.array([self._objective(x) for x in X])
            mask = current_val < pbest_val
            pbest[mask] = X[mask]
            pbest_val[mask] = current_val[mask]

            gbest_idx = np.argmin(pbest_val)
            if pbest_val[gbest_idx] < gbest_val:
                gbest = pbest[gbest_idx].copy()
                gbest_val = pbest_val[gbest_idx]

            print(f"Iteration {t+1}: Best Fitness = {gbest_val:.4f}")
            history.append(gbest_val)

        best_mass = self.mass_model.predict_values(gbest.reshape(1, -1))[0][0]
        best_stress = self.stress_model.predict_values(gbest.reshape(1, -1))[0][0]

        print("\n=== 最优设计结果 ===")
        for i, val in enumerate(gbest):
            print(f"p{i + 1} = {val:.4f}")
        print(f"预测质量: {best_mass:.4f}")
        print(f"预测应力: {best_stress:.4f}")

        self._plot_convergence(history)

    def _plot_convergence(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(history) + 1), history, label='Best Fitness So Far')
        plt.axhline(y=history[-1], color='r', linestyle='--',
                    label=f'Convergence: {history[-1]:.4f}')
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title("PSO Convergence Plot")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.work_dir, 'pso.svg'), bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    pso_optimizer = KrigingPSOOptimizer()
    pso_optimizer.run_pso()
