import os
import sqlite3
import datetime
import numpy as np
import pandas as pd
from scipy.stats import qmc
import random

class AnsysAutomation:
    def __init__(self, db_path='data_ansys.db'):
        self.db_path = db_path
        self.template_file = ""
        self.template_content = ""
        self.lower_bounds = []
        self.upper_bounds = []
        self.n_variables = 0
        self.ansys_exe = ""
        self.work_dir = ""
        self.result_mass_file = ""
        self.result_stress_file = ""

        self._read_database()

    def _read_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 读取 ANSYS 执行路径与工作路径
        cursor.execute("SELECT path FROM path WHERE id=1")
        self.ansys_exe = cursor.fetchone()[0]

        cursor.execute("SELECT path FROM path WHERE id=2")
        self.work_dir = cursor.fetchone()[0]

        # 输入文件路径（模板）
        cursor.execute("SELECT path FROM path WHERE id=3")
        self.template_file = cursor.fetchone()[0]

        if not os.path.exists(self.template_file):
            raise FileNotFoundError("模板文件不存在")

        with open(self.template_file, 'r', encoding='utf-8') as f:
            self.template_content = f.read()

        # 读取变量上下限
        cursor.execute("SELECT lower_bound, upper_bound FROM variables")
        rows = cursor.fetchall()
        self.lower_bounds = [row[0] for row in rows]
        self.upper_bounds = [row[1] for row in rows]
        self.n_variables = len(rows)

        # 读取目标和约束结果文件路径
        cursor.execute("SELECT work_dir FROM aim_function")
        self.result_mass_file = cursor.fetchone()[0]

        cursor.execute("SELECT work_dir FROM constraint_function")
        self.result_stress_file = cursor.fetchone()[0]

        conn.close()

    def generate_lhs_samples(self, n_samples=5):
        sampler = qmc.LatinHypercube(d=self.n_variables)
        sample = sampler.random(n=n_samples)
        scaled_sample = qmc.scale(sample, self.lower_bounds, self.upper_bounds)
        return scaled_sample

    def write_modified_input(self, values):
        modified = self.template_content
        for j, val in enumerate(values):
            tag = f"A{j+1}"
            formatted_val = f"{val:.4f}".ljust(8)
            modified = modified.replace(tag, formatted_val)
        mod_path = self.template_file.replace('.mac', '_modified.mac')
        with open(mod_path, 'w', encoding='utf-8') as f:
            f.write(modified)
        return mod_path

    def run_ansys(self, input_file):
        output_log = os.path.join(self.work_dir, 'ansys_output.log')
        command = f"\"{self.ansys_exe}\" -b -p ane3fl -j ans -dir {self.work_dir} -i {input_file} -o {output_log}"
        os.system(command)

    def read_output_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"找不到文件: {filepath}")
            return None
        with open(filepath, 'r') as f:
            line = f.readline()
            try:
                return float(line.strip())
            except ValueError:
                print(f"无法读取数值: {filepath}")
                return None

    def save_results_to_csv(self, results, csv_path):
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)

    def run(self, n_samples=5):
        results = []
        samples = self.generate_lhs_samples(n_samples)

        for i, values in enumerate(samples):
            print(f"\n🔧 样本{i+1} 正在处理...")
            design_vars = {f"p{j+1}": val for j, val in enumerate(values)}

            input_file = self.write_modified_input(values)
            self.run_ansys(input_file)

            mass = self.read_output_file(self.result_mass_file)
            stress = self.read_output_file(self.result_stress_file)

            print(f"Design Variables: {design_vars}")
            print(f"Mass: {mass} | Stress: {stress}")
            print(datetime.datetime.now().strftime("%H:%M:%S"))

            result = {**design_vars, "Mass": mass, "Stress": stress}
            results.append(result)

        csv_path = os.path.join(self.work_dir, "results_summary.csv")
        self.save_results_to_csv(results, csv_path)
        print(f"\n所有结果已保存至: {csv_path}")


    def split_train_val(self, csv_path, n_train=10, n_val=5):
        df = pd.read_csv(csv_path)

        if len(df) < n_train + n_val:
            raise ValueError(f"数据量不足，当前只有 {len(df)} 条记录，无法划分出 {n_train} 个训练点和 {n_val} 个验证点")

        # 随机抽样索引
        indices = list(df.index)
        random.shuffle(indices)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]

        train_df = df.loc[train_indices].reset_index(drop=True)
        val_df = df.loc[val_indices].reset_index(drop=True)

        # 保存
        train_path = os.path.join(os.path.dirname(csv_path), "krg_train_data.csv")
        val_path = os.path.join(os.path.dirname(csv_path), "krg_val_data.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        print(f"训练集已保存至: {train_path}")
        print(f"验证集已保存至: {val_path}")
if __name__ == "__main__":
    runner = AnsysAutomation()
    runner.run(n_samples=20)  # 假设共生成20个样本点
    result_csv = os.path.join(runner.work_dir, "results_summary.csv")
    runner.split_train_val(result_csv, n_train=15, n_val=5)

