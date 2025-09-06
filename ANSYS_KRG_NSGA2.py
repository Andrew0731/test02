import os
import sqlite3
import numpy as np
import random
import joblib
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


class KrigingNSGA2Optimizer:
    def __init__(self, db_path='data_ansys.db', log_callback=None):
        self.db_path = db_path
        self.work_dir = self._get_work_dir()
        self.krg_dir = os.path.join(self.work_dir, "krg_models")
        os.makedirs(self.krg_dir, exist_ok=True)
        self.log_callback = log_callback

        self.mass_model = None
        self.stress_model = None
        self.bounds = []

        self._load_models()
        self._load_variable_bounds()
        self._setup_deap()

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

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

    def _setup_deap(self):
        if not hasattr(creator, "FitnessMin"):
            # 多目标优化，两个目标都是最小化
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._objective)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[b[0] for b in self.bounds],
                              up=[b[1] for b in self.bounds], eta=20.0)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=[b[0] for b in self.bounds],
                              up=[b[1] for b in self.bounds], eta=20.0, indpb=1.0/len(self.bounds))
        self.toolbox.register("select", tools.selNSGA2)

    def _generate_individual(self):
        return [random.uniform(low, high) for (low, high) in self.bounds]

    def _objective(self, individual):
        x = np.array(individual).reshape(1, -1)
        mass = self.mass_model.predict_values(x)[0][0]
        stress = self.stress_model.predict_values(x)[0][0]
        return mass, stress

    def run_nsga2(self, n_gen=100, pop_size=100):
        pop = self.toolbox.population(n=pop_size)
        # 初始化评价并分配pareto等级
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        pop = self.toolbox.select(pop, len(pop))

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda fits: np.mean([f[0] for f in fits])), stats.register("avg2", lambda fits: np.mean([f[1] for f in fits]))
        stats.register("min", lambda fits: np.min([f[0] for f in fits])), stats.register("min2", lambda fits: np.min([f[1] for f in fits]))

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals", "avg_mass", "avg_stress", "min_mass", "min_stress"]

        for gen in range(1, n_gen + 1):
            offspring = algorithms.varAnd(pop, self.toolbox, cxpb=0.9, mutpb=0.1)
            fitnesses = list(map(self.toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
            pop = self.toolbox.select(pop + offspring, pop_size)

            fits = [ind.fitness.values for ind in pop]
            avg_mass = np.mean([f[0] for f in fits])
            avg_stress = np.mean([f[1] for f in fits])
            min_mass = np.min([f[0] for f in fits])
            min_stress = np.min([f[1] for f in fits])

            logbook.record(gen=gen, evals=len(offspring), avg_mass=avg_mass, avg_stress=avg_stress,
                           min_mass=min_mass, min_stress=min_stress)

            self._log(f"Gen {gen}: Avg Mass={avg_mass:.4f}, Avg Stress={avg_stress:.4f}, Min Mass={min_mass:.4f}, Min Stress={min_stress:.4f}")

        # 输出Pareto前沿个体
        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

        self._log("\n=== Pareto Front Solutions ===")
        for i, ind in enumerate(pareto_front):
            mass, stress = ind.fitness.values
            vars_str = ", ".join(f"p{j+1}={val:.4f}" for j, val in enumerate(ind))
            self._log(f"Solution {i+1}: {vars_str}, Mass={mass:.4f}, Stress={stress:.4f}")

        # 绘制Pareto前沿图
        masses = [ind.fitness.values[0] for ind in pareto_front]
        stresses = [ind.fitness.values[1] for ind in pareto_front]

        plt.figure(figsize=(8,6))
        plt.scatter(masses, stresses, c='red', label='Pareto Front')
        plt.xlabel('Mass')
        plt.ylabel('Stress')
        plt.title('NSGA-II Pareto Front')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.work_dir, "nsga2_pareto_front.svg"), bbox_inches='tight')
        plt.show()

if __name__ =="__main__":
    nsga2_optimizer = KrigingNSGA2Optimizer()
    nsga2_optimizer.run_nsga2()