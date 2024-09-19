import warnings
import pandas as pd
import numpy as np
import random

# Used for splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
# Used for training a logistic regression classifier
from sklearn.linear_model import LogisticRegression
# Used for calculating accuracy as the fitness score
from sklearn.metrics import accuracy_score
# Used for implementing the Genetic Algorithm framework
from deap import creator, base, tools, algorithms

def avg(l):
    return sum(l) / float(len(l))

dataset_path = r'C:\Users\19180\PycharmProjects\metaHeuristic\diabetes.csv'

def load_dataset(dataset_path):
    # Load the dataset from path
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def evaluate_individual(individual, X_train, X_test, y_train, y_test):
    # Evaluate the fitness of an individual (feature subset)
    if individual.count(0) != len(individual):
        # Get the indices of selected features (where individual[index] == 0)
        selected_feature_indices = [index for index in range(len(individual)) if individual[index] == 0]

        # Drop the columns corresponding to unselected features
        X_train_subset = X_train.drop(X_train.columns[selected_feature_indices], axis=1)
        X_test_subset = X_test.drop(X_test.columns[selected_feature_indices], axis=1)

        # Suppress ConvergenceWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Train a logistic regression classifier on the subset of features
            clf = LogisticRegression(max_iter=1000)  # Increase max_iter to avoid convergence warnings
            clf.fit(X_train_subset, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test_subset)

        # Calculate accuracy as the fitness score
        fitness_score = accuracy_score(y_test, y_pred)
        return fitness_score,
    else:
        # If no features are selected, return a fitness score of 0
        return 0,

def genetic_algorithm(X_train, X_test, y_train, y_test, n_population, n_generations):
    # Create the DEAP types for fitness and individuals
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Create the DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(X_train.columns))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=5)

    # Initialize the population and hall of fame
    population = toolbox.population(n=n_population)
    hall_of_fame = tools.HallOfFame(n_population * n_generations)

    # Statistics to be computed for the population
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm
    # Mutation probability = 0.05
    # Crossover probability = 0.6
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.6, mutpb=0.05,
                                              ngen=n_generations, stats=stats, halloffame=hall_of_fame, verbose=True)

    return hall_of_fame

def get_best_individual(hall_of_fame, X_train, X_test, y_train, y_test):
    # Find the best individual from the hall of fame
    best_fitness_tuple, best_individual = hall_of_fame[0].fitness.values, hall_of_fame[0]
    best_fitness = best_fitness_tuple[0]  # Extract the accuracy value from the tuple
    best_header = [list(X_train)[i] for i in range(len(best_individual)) if best_individual[i] == 1]
    return best_fitness, best_individual, best_header

def main(dataset_path, n_population=50, n_generations_list=[100, 500, 1000, 2000]):
    # Load the dataset
    X, y = load_dataset(dataset_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    for n_generations in n_generations_list:
        # Apply the genetic algorithm
        hall_of_fame = genetic_algorithm(X_train, X_test, y_train, y_test, n_population, n_generations)

        # Get the best individual from the hall of fame
        best_fitness, best_individual, best_header = get_best_individual(hall_of_fame, X_train, X_test, y_train, y_test)

        # Print the results for the current run
        print(f'Output after {n_generations} runs:')
        print('Best Accuracy: \t', best_fitness)
        print('Number of Features in Subset: \t', best_individual.count(1))
        print('Individual: \t\t', best_individual)
        print('Feature Subset: \t', best_header)
        print('\n\n')

if __name__ == '__main__':
    dataset_path = r'C:\Users\19180\PycharmProjects\metaHeuristic\diabetes.csv'

    # Genetic algorithm parameters
    population_size = 50
    generations_list = [100, 500, 1000, 2000]

    # Sessize alınacak uyarılar
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")

    # Run the genetic algorithm for feature selection with different numbers of generations
    main(dataset_path, n_population=population_size, n_generations_list=generations_list)
