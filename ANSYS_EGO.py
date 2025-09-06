import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from smt.surrogate_models import KRG
from scipy.optimize import minimize
from scipy.stats import norm
import time
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

def prepare_initial_data(results):
    X_init = []
    Y_mass_init = []
    Y_stress_init = []

    for item in results:
        if item["Mass"] is not None and item["Stress"] is not None:
            try:
                x = [item["p1"], item["p2"], item["p3"],item["p4"]]
                y_mass = float(item["Mass"])
                y_stress = float(item["Stress"])
                X_init.append(x)
                Y_mass_init.append([y_mass])
                Y_stress_init.append([y_stress])
            except ValueError:
                continue

    return np.array(X_init), np.array(Y_mass_init), np.array(Y_stress_init)


def replace_design_variables(input_filepath, output_filepath, var_values):
    with open(input_filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    for key, value in var_values.items():
        pattern = r"(" + key + r"\s*=\s*)var" + key[-1]
        replacement = r"\g<1>" + str(value)
        content = re.sub(pattern, replacement, content)
    with open(output_filepath, 'w', encoding='utf-8') as file:
        file.write(content)


def run_ansys(ansys_exe, work_dir, input_file, output_log):
    command = f"\"{ansys_exe}\" -b -p ane3fl -j ans -dir {work_dir} -i {input_file} -o {output_log}"
    return os.system(command)


def read_output_file(filepath):
    """改进：直接返回浮点数，增强错误处理"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            try:
                return float(content)  # 直接转换为数值
            except ValueError:
                print(f"错误：{filepath} 包含非数值内容: {content}")
                return None
    print(f"错误：{filepath} 不存在")
    return None


def generate_lhs_samples(n_samples):
    # 定义变量范围
    bounds = {
        "p1": (14 * 0.8, 16 * 1.2),
        "p2": (16 * 0.8, 14 * 1.2),
        "p3": (12 * 0.8, 12 * 1.2),
        "p4": (15 * 0.8, 12 * 1.2)
    }
    keys = list(bounds.keys())
    l_bounds = [bounds[k][0] for k in keys]
    u_bounds = [bounds[k][1] for k in keys]

    sampler = qmc.LatinHypercube(d=4)  # 4个变量
    sample = sampler.random(n=n_samples)  # 生成 n 个样本
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

    # 将样本整理成字典列表 [{p1: ..., p2: ..., p3: ... ,p4:...}, ...]
    all_samples = []
    for row in scaled_sample:
        sample_dict = {key: round(val, 4) for key, val in zip(keys, row)}
        all_samples.append(sample_dict)

    return all_samples


def build_kriging(X, Y):
    krg = KRG(print_global=False)
    krg.set_training_values(X, Y)
    krg.train()
    return krg


def constrained_ei(x, krg_mass, krg_stress, stress_limit, f_min, existing_X=None, tol=1e-4):
    x = np.array(x).reshape(1, -1)

    if existing_X is not None:
        if np.any(np.linalg.norm(existing_X - x, axis=1) < tol):
            return 1e6  # 避免重复采样

    # 质量模型预测
    mu_m = krg_mass.predict_values(x)[0][0]
    sigma_m = max(np.sqrt(krg_mass.predict_variances(x)[0][0]), 1e-6)  # 避免零方差

    # 应力模型预测
    mu_s = krg_stress.predict_values(x)[0][0]
    sigma_s = max(np.sqrt(krg_stress.predict_variances(x)[0][0]), 1e-6)  # 避免零方差

    # 期望改进（EI）计算
    if sigma_m > 1e-6:
        z = (f_min - mu_m) / sigma_m
        ei = (f_min - mu_m) * norm.cdf(z) + sigma_m * norm.pdf(z)
    else:
        ei = 0.0  # 方差过小时无改进潜力

    # 可行性概率（应力 <= 限制）
    if sigma_s > 1e-6:
        p_feas = norm.cdf((stress_limit - mu_s) / sigma_s)
    else:
        p_feas = 1.0 if (mu_s <= stress_limit) else 0.0  # 方差过小时确定性判断

    return -ei * p_feas  # 最小化负EI×可行性概率


def run_ansys_model(x_array, original_file, modified_file, ansys_exe, work_dir, output_log, mass_file, stress_file):
    design_vars = {f"p{i + 1}": round(val, 4) for i, val in enumerate(x_array)}
    replace_design_variables(original_file, modified_file, design_vars)
    ret = run_ansys(ansys_exe, work_dir, modified_file, output_log)

    if ret != 0:
        print("ANSYS运行失败，错误码:", ret)
        return None, None

    mass = read_output_file(mass_file)
    stress = read_output_file(stress_file)

    return mass, stress  # 直接返回数值或None


def run_ego(bounds, X_init, Y_mass_init, Y_stress_init,
            stress_limit, max_iter, original_file, modified_file,
            ansys_exe, work_dir, output_log, mass_file, stress_file):
    X = X_init.copy()
    Y_mass = Y_mass_init.copy()
    Y_stress = Y_stress_init.copy()

    # ------------------- 改进1：初始化最佳解（包含初始样本） -------------------
    feasible_mask = Y_stress.flatten() <= stress_limit
    if np.any(feasible_mask):
        initial_best_idx = np.argmin(Y_mass[feasible_mask])
        best_mass = Y_mass[feasible_mask][initial_best_idx][0]
        best_stress = Y_stress[feasible_mask][initial_best_idx][0]
        best_design = X[feasible_mask][initial_best_idx]
        print(f"初始最佳解（来自初始样本）: Mass={best_mass:.4f}, Stress={best_stress:.4f}")
    else:
        best_mass, best_stress, best_design = None, None, None

    history = []  # 记录每一步的最佳质量（可行解优先，否则取最小质量）
    plt.figure(figsize=(10, 6))

    for i in range(max_iter):
        print(f"\n--- EGO 迭代 {i + 1}/{max_iter} ---")

        # 更新当前最佳解（考虑新加入的样本）
        current_feasible_mask = Y_stress.flatten() <= stress_limit
        if np.any(current_feasible_mask):
            current_best_idx = np.argmin(Y_mass[current_feasible_mask])
            f_min = Y_mass[current_feasible_mask][current_best_idx][0]
        else:
            f_min = np.min(Y_mass)  # 无可行解时取最小质量（鼓励探索）

        # 构建Kriging模型
        krg_mass = build_kriging(X, Y_mass)
        krg_stress = build_kriging(X, Y_stress)

        # ------------------- 改进2：增强多起点优化（从3次增加到10次） -------------------
        candidates = []
        for _ in range(10):  # 更多起点提高全局搜索能力
            x0 = np.array([np.random.uniform(low, high) for (low, high) in bounds.values()])
            res = minimize(
                lambda x: constrained_ei(x, krg_mass, krg_stress, stress_limit, f_min, X),
                x0=x0,
                bounds=list(bounds.values()),
                method='L-BFGS-B',
                options={'maxiter': 300}
            )
            if res.success:
                candidates.append((res.x, res.fun))

        if not candidates:
            print("优化失败，使用随机采样")
            x_new = np.array([np.random.uniform(low, high) for (low, high) in bounds.values()])
        else:
            candidates.sort(key=lambda x: x[1])  # 按目标函数值排序
            x_new = candidates[0][0]  # 选择最优的候选点

        x_new_rounded = np.round(x_new.reshape(1, -1), 4)

        # 运行ANSYS仿真
        mass_new, stress_new = run_ansys_model(
            x_new_rounded[0], original_file, modified_file,
            ansys_exe, work_dir, output_log, mass_file, stress_file
        )

        if mass_new is None or stress_new is None:
            print("仿真结果无效，跳过本次采样")
            continue

        # 更新数据集
        X = np.vstack((X, x_new_rounded))
        Y_mass = np.vstack((Y_mass, [[mass_new]]))
        Y_stress = np.vstack((Y_stress, [[stress_new]]))

        # 更新全局最佳解
        if stress_new <= stress_limit:
            if (best_design is None) or (mass_new < best_mass):
                best_mass = mass_new
                best_stress = stress_new
                best_design = x_new_rounded[0]
                print(f"找到新的可行解（当前最佳）: Mass={best_mass:.4f}, Stress={best_stress:.4f}")

        # 记录历史（优先记录可行解的最佳质量，无可行解时记录最小质量）
        current_best = best_mass if best_mass is not None else np.min(Y_mass)
        history.append(current_best)

        # 实时更新收敛曲线
        plt.clf()
        plt.plot(range(1, len(history) + 1), history, 'bo-', alpha=0.7)
        plt.xlabel('迭代次数')
        plt.ylabel('最佳质量')
        plt.title(f'EGO优化进度（迭代 {i + 1}）')
        plt.grid(True)
        plt.pause(0.1)


    return X, Y_mass, Y_stress, best_design, best_mass, best_stress, history


def main():
    # ------------------- 配置参数（根据实际路径修改） -------------------
    original_file = r"F:\apdl_temp\goujia\goujia_test2.mac"
    modified_file = r"F:\apdl_temp\goujia\goujia_test2_modified.mac"
    mass_file = r"F:\apdl_temp\goujia\mass_file.txt"
    stress_file = r"F:\apdl_temp\goujia\stress_file.txt"

    ansys_exe = r"F:\soft\ansys19.2\ANSYS Inc\v192\ANSYS\bin\winx64\ANSYS192.exe"
    work_dir = r"F:\apdl_temp\goujia"
    data_dir = r"F:\apdl_temp\goujia\data"
    output_log = r"F:\apdl_temp\goujia\output.log"

    bounds = {
        "p1": (14 * 0.8, 14 * 1.2),
        "p2": (16 * 0.8, 16 * 1.2),
        "p3": (12 * 0.8, 12 * 1.2),
        "p4": (15 * 0.8, 15 * 1.2)
    }

    # ------------------- 改进3：增加初始样本量（从5增加到10） -------------------
    num_initial = 7  # 初始样本量，建议10-15个
    design_var_list = generate_lhs_samples(num_initial)
    results = []

    print("=== 初始采样阶段 ===")
    for i, design_vars in enumerate(design_var_list, 1):
        print(f"样本 {i}: p1={design_vars['p1']:.4f}, p2={design_vars['p2']:.4f}, p3={design_vars['p3']:.4f}, p4={design_vars['p4']:.4f}")
        replace_design_variables(original_file, modified_file, design_vars)
        ret = run_ansys(ansys_exe, work_dir, modified_file, output_log)

        if ret != 0:
            print(f"样本 {i} 仿真失败，错误码: {ret}")
            results.append({**design_vars, "Mass": None, "Stress": None})
            continue

        mass = read_output_file(mass_file)
        stress = read_output_file(stress_file)
        results.append({
            "p1": design_vars['p1'],
            "p2": design_vars['p2'],
            "p3": design_vars['p3'],
            "p4": design_vars['p4'],
            "Mass": mass,
            "Stress": stress
        })
        print(f"结果 - 质量: {mass:.4f}, 应力: {stress:.4f}")

    # 准备初始数据
    X_init, Y_mass_init, Y_stress_init = prepare_initial_data(results)

    # EGO优化参数
    stress_limit = 322.0  # 应力约束上限
    max_iter = 13  # 优化迭代次数

    # 运行EGO优化
    X_final, Y_mass_final, Y_stress_final, best_design, best_mass, best_stress, history = run_ego(
        bounds, X_init, Y_mass_init, Y_stress_init,
        stress_limit, max_iter, original_file, modified_file,
        ansys_exe, work_dir, output_log, mass_file, stress_file
    )

    # 结果输出
    print("\n=== 最终优化结果 ===")
    if best_design is not None:
        print("最优设计变量:")
        print(f"p1 = {best_design[0]:.4f}")
        print(f"p2 = {best_design[1]:.4f}")
        print(f"p3 = {best_design[2]:.4f}")
        print(f"p4 = {best_design[3]:.4f}")
        print(f"最小质量: {best_mass:.4f}")
        print(f"对应应力: {best_stress:.4f}")
    else:
        print("未找到可行解！")

    csv_file = os.path.join(data_dir, 'optimization_history.csv')
    file_exists = os.path.exists(csv_file)  # 提前检查文件是否存在

    with open(csv_file, 'a', newline='') as csvfile:  # 改为追加模式 'a'
        writer = csv.writer(csvfile)
        if not file_exists:  # 仅在文件不存在时写入表头
            writer.writerow(['p1', 'p2', 'p3', 'p4', 'Mass', 'Stress', 'feasible'])

        # 写入当前迭代的数据（无论文件是否存在，直接追加）
        for idx in range(len(X_final)):
            feasible = Y_stress_final[idx][0] <= stress_limit
            writer.writerow([
                X_final[idx, 0],
                X_final[idx, 1],
                X_final[idx, 2],
                X_final[idx, 3],
                Y_mass_final[idx, 0],
                Y_stress_final[idx, 0],
                feasible
            ])

    print("\n优化数据已保存到 optimization_history.csv")
    print("收敛曲线已保存到 ego_convergence.png")


if __name__ == "__main__":
    time0 = time.time()
    main()
    time1 = time.time()
    print(f"优化时间: {time1 - time0}")