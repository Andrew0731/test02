import os
import sqlite3
import numpy as np
import random
import joblib
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

class KrigingGAOptimizer:
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
        self._setup_deap()

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
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._objective)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _generate_individual(self):
        return [random.uniform(low, high) for (low, high) in self.bounds]

    def _is_within_bounds(self, individual):
        return all(low <= x <= high for x, (low, high) in zip(individual, self.bounds))

    def _objective(self, individual):
        if not self._is_within_bounds(individual):
            return 1e8,

        x = np.array(individual).reshape(1, -1)
        mass = self.mass_model.predict_values(x)[0][0]
        stress = self.stress_model.predict_values(x)[0][0]
        penalty = 1e6 * max(0, stress - 312) ** 2
        return mass + penalty,

    def run_ga(self, n_gen=100, pop_size=100):
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)

        best_values = []
        for gen in range(n_gen):
            pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.2, ngen=1,
                                               stats=stats, halloffame=hof, verbose=False)
            best_ind = tools.selBest(pop, k=1)[0]
            best_value = best_ind.fitness.values[0]
            best_values.append(best_value)
            print(f"Generation {gen + 1}: Best Fitness = {best_value:.4f}")

        best = hof[0]
        best_mass = self.mass_model.predict_values(np.array(best).reshape(1, -1))[0][0]
        best_stress = self.stress_model.predict_values(np.array(best).reshape(1, -1))[0][0]

        print("\n=== 最优设计结果 ===")
        for i, val in enumerate(best):
            print(f"p{i + 1} = {val:.4f}")
        print(f"预测质量: {best_mass:.4f}")
        print(f"预测应力: {best_stress:.4f}")

        self._plot_convergence(best_values)

    def _plot_convergence(self, best_values):
        generations = range(1, len(best_values) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_values, label="Best Fitness So Far")
        convergence_value = best_values[-1]
        plt.axhline(y=convergence_value, color='r', linestyle='--',
                    label=f'Convergence Value: {convergence_value:.4f}')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Genetic Algorithm Iteration Plot (Best Value So Far)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.work_dir, 'ga.svg'), bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    GA_Optimizer = KrigingGAOptimizer()
    GA_Optimizer.run_ga()

