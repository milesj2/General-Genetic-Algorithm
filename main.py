from random import random, randint, uniform, seed, gauss
from copy import deepcopy
from math import cos, pi, exp, sqrt, e


# GENERAL
NUM_EXPERIMENTS = 1
MAX_RUNS = 10
NUM_GENERATIONS = 50
POP_SIZE = 500

# FITNESS FUNCTIONS
F_COUNTING_ONES_INT = 0
F_COUNTING_ONES_MAX = 1
F_COUNTING_ONES_MIN = 2
F_RASTRIGINS = 3
F_RASTRIGINS_LONG = 4
F_SECOND_MINIMISATION = 5
FUNCTION = F_RASTRIGINS_LONG
MINIMISATION = True

# GENE VALUES
GENE_SIZE = 50
GENE_INT_VALUES = False
GENE_MAX_VAL = 1
GENE_MIN_VAL = 0

# HEURISTICS
HEURISTIC_POP = False
HEURISTIC_PROPORTION = int(POP_SIZE * 0.125)

# SELECTION
TOURNAMENT = "TOURNAMENT"
ROULETTE = "ROULETTE"
ELITE = "ELITE"
RANK = "RANK"
SELECTION = TOURNAMENT
TOURNAMENT_SIZE = 10

# MUTATION
MUTATION_STEP = 3
MUTATION_PERCENTAGE = 4
GAUSSIAN_MUTATION = False
MUTATION_MEAN = 1
MUTATION_STD_DEVIATION = 2

# CROSSOVER
CROSSOVER_SINGLE = "SINGLE"
CROSSOVER_DOUBLE = "DOUBLE"
CROSSOVER_UNIFORM = "UNIFORM"
CROSSOVER_TYPE = CROSSOVER_UNIFORM
CROSSOVER_PERCENTAGE = 100

# ELITISM
ELITISM = True
ELITE_PROPORTION = int(POP_SIZE * 0.125)

# IMMIGRANTS
IMMIGRANTS = False
IMMIGRANT_PROPORTION = int(POP_SIZE * 0.05)
IMMIGRANT_END_GENERATION = int(NUM_GENERATIONS * 0.6)

# MISC
DATA_PATH = os.getcwd()
HALF_CHANCE = 50
DEBUG = False
PRINT_FINAL_POPULATION = False


def initialise_problem():
    global GENE_MAX_VAL, GENE_MIN_VAL, GENE_SIZE
    global MINIMISATION, GENE_INT_VALUES, MUTATION_STEP
    global MUTATION_MEAN, MUTATION_STD_DEVIATION

    if FUNCTION == F_COUNTING_ONES_INT:
        MINIMISATION = False
        MUTATION_STEP = 1
        GENE_INT_VALUES = True
    elif FUNCTION == F_COUNTING_ONES_MAX:
        MINIMISATION = False
    elif FUNCTION == F_RASTRIGINS or FUNCTION == F_RASTRIGINS_LONG:
        GENE_MAX_VAL = 5.12
        GENE_MIN_VAL = -5.12
        if FUNCTION == F_RASTRIGINS:
            GENE_SIZE = 2
        else:
            GENE_SIZE = 10
    elif FUNCTION == F_SECOND_MINIMISATION:
        GENE_SIZE = 10
        GENE_MAX_VAL = 32.0
        GENE_MIN_VAL = -32.0
        MUTATION_STEP = 1
        MUTATION_MEAN = 0.4
        MUTATION_STD_DEVIATION = 0.5


initialise_problem()


class Individual:
    def __init__(self):
        self.gene = [0.00] * GENE_SIZE
        self.fitness = 0


population = [Individual() for i in range(POP_SIZE)]
elite_individuals = [Individual() for i in range(ELITE_PROPORTION)]


def main(run):
    global elite_individuals
    initialise_problem()
    initialise_population()
    calculate_fitness()
    for generation in range(NUM_GENERATIONS):
        save_data(run, generation)
        if ELITISM:
            calculate_fitness()
            elite_individuals = deepcopy(sort_population()[:ELITE_PROPORTION])
        crossover_population()
        mutation_population()
        calculate_fitness()
        select_new_generation(elite_individuals, generation)
    calculate_fitness()


def initialise_population():
    global population
    for individual in population:
        for i in range(GENE_SIZE):
            if GENE_INT_VALUES:
                individual.gene[i] = randint(GENE_MIN_VAL, GENE_MAX_VAL)
            else:
                individual.gene[i] = uniform(GENE_MIN_VAL, GENE_MAX_VAL)

    val_range = GENE_MAX_VAL - GENE_MIN_VAL
    if HEURISTIC_POP:
        for i in range(HEURISTIC_PROPORTION):
            value = (val_range/(POP_SIZE/2) * (i+1)) - (0 - GENE_MIN_VAL)
            population[i].gene = [value] * GENE_SIZE


def crossover_population():
    if CROSSOVER_TYPE == CROSSOVER_SINGLE:
        single_point_crossover()
    elif CROSSOVER_TYPE == CROSSOVER_DOUBLE:
        double_point_crossover()
    elif CROSSOVER_TYPE == CROSSOVER_UNIFORM:
        uniform_crossover()


def single_point_crossover():
    donor_index = -2
    while donor_index < POP_SIZE - 2:
        donor_index += 2
        if not chance_of(CROSSOVER_PERCENTAGE):
            continue
        cross_position = randint(0, GENE_SIZE - 1)

        donor = population[donor_index]
        recipient = population[donor_index + 1]

        for i in range(cross_position, GENE_SIZE):
            temp_gene = recipient.gene[i]
            recipient.gene[i] = donor.gene[i]
            donor.gene[i] = temp_gene


def double_point_crossover():
    donor_index = -2
    while donor_index < POP_SIZE - 2:
        donor_index += 2
        if not chance_of(CROSSOVER_PERCENTAGE):
            continue
        first_position = randint(0, GENE_SIZE - 1)
        second_position = randint(first_position, GENE_SIZE)

        donor = population[donor_index]
        recipient = population[donor_index + 1]

        temp_array = deepcopy(donor.gene[first_position:second_position])
        donor.gene[first_position:second_position] = deepcopy(recipient.gene[first_position:second_position])
        recipient.gene[first_position:second_position] = deepcopy(temp_array)


def uniform_crossover():
    donor_index = -2
    while donor_index < POP_SIZE - 2:
        donor_index += 2
        if not chance_of(CROSSOVER_PERCENTAGE):
            continue
        parent1 = population[donor_index]
        parent2 = population[donor_index + 1]

        new_individual_1 = Individual()
        new_individual_2 = Individual()

        for i in range(GENE_SIZE):
            if chance_of(HALF_CHANCE):
                new_individual_1.gene[i] = parent1.gene[i]
                new_individual_2.gene[i] = parent2.gene[i]
            else:
                new_individual_2.gene[i] = parent1.gene[i]
                new_individual_1.gene[i] = parent2.gene[i]
        population[donor_index] = deepcopy(new_individual_1)
        population[donor_index + 1] = deepcopy(new_individual_2)


def mutation_population():
    for individual in population:
        for i in range(GENE_SIZE):
            if not chance_of(MUTATION_PERCENTAGE):
                continue
            if GENE_INT_VALUES:
                mutation_amount = randint(0, int(MUTATION_STEP))
            else:
                if GAUSSIAN_MUTATION:
                    mutation_amount = trunc_gauss(0, MUTATION_STEP, -MUTATION_STEP, MUTATION_STEP)
                else:
                    mutation_amount = uniform(-MUTATION_STEP, MUTATION_STEP)
            individual.gene[i] += mutation_amount

            if individual.gene[i] > GENE_MAX_VAL:
                individual.gene[i] = GENE_MAX_VAL
            elif individual.gene[i] < GENE_MIN_VAL:
                individual.gene[i] = GENE_MIN_VAL


def select_new_generation(elite, generation):
    if SELECTION == TOURNAMENT:
        tournament_selection()
    elif SELECTION == ROULETTE:
        roulette_selection()
    elif SELECTION == ELITE:
        elite_selection()
    elif SELECTION == RANK:
        rank_selection()

    if ELITISM:
        population[:ELITE_PROPORTION] = deepcopy(elite)
    if IMMIGRANTS and generation < IMMIGRANT_END_GENERATION:
        for individual in range(IMMIGRANT_PROPORTION, POP_SIZE - IMMIGRANT_PROPORTION - 1):
            for i in range(GENE_SIZE):
                if GENE_INT_VALUES:
                    population[individual].gene[i] = randint(GENE_MIN_VAL, GENE_MAX_VAL)
                else:
                    population[individual].gene[i] = uniform(GENE_MIN_VAL, GENE_MAX_VAL)


def tournament_selection():
    global population

    new_generation = []
    for i in range(POP_SIZE):
        highest = population[randint(0, POP_SIZE-1)]
        lowest = highest

        for j in range(1, TOURNAMENT_SIZE):
            challenger = population[randint(0, POP_SIZE-1)]
            if challenger.fitness > highest.fitness:
                highest = challenger
            elif challenger.fitness < lowest.fitness:
                lowest = challenger

        if MINIMISATION:
            new_generation.append(deepcopy(lowest))
        else:
            new_generation.append(deepcopy(highest))
    population = deepcopy(new_generation)


def elite_selection():
    global population
    sorted_population = sort_population()

    elite_size = int(POP_SIZE/8)

    new_generation = elite_individuals

    for i in range(int(POP_SIZE/2)):
        new_generation.append(sorted_population[randint(0, elite_size)])
    while len(new_generation) != POP_SIZE:
        new_generation.append(sorted_population[randint(0, POP_SIZE - 1)])

    population = deepcopy(new_generation)


def rank_selection():
    global population
    new_generation = []

    sorted_pop = sort_population()

    for i in range(POP_SIZE):
        pick = randint(0, POP_SIZE-1)
        current = 0
        for j in range(POP_SIZE):
            current += j
            if current >= pick:
                new_generation.append(sorted_pop[j])
    population = deepcopy(new_generation)


def roulette_selection():
    global population
    new_generation = []
    sorted_pop = sort_population()

    if MINIMISATION:
        append = 0
        if sorted_pop[0].fitness < 0:
            append = -sorted_pop[0].fitness
        for i in population:
            if i.fitness == 0:
                continue
            i.fitness = 1/i.fitness + append

    max_probability = sum(individual.fitness for individual in population)

    for i in range(POP_SIZE):
        pick = uniform(0, max_probability)
        current = 0
        for individual in sorted_pop:
            current += individual.fitness
            if current > pick:
                new_generation.append(deepcopy(individual))
                break



    population = deepcopy(new_generation)
    calculate_fitness()


def calculate_fitness():
    if FUNCTION in [F_COUNTING_ONES_INT, F_COUNTING_ONES_MIN, F_COUNTING_ONES_MAX]:
        sum_fitness()
    elif FUNCTION == F_RASTRIGINS or FUNCTION == F_RASTRIGINS_LONG:
        hagrid_fitness()
    elif FUNCTION == F_SECOND_MINIMISATION:
        second_minimisation()


def sum_fitness():
    for individual in population:
        individual.fitness = 0
        for gene in individual.gene:
            individual.fitness += gene


def hagrid_fitness():
    for individual in population:
        individual.fitness = 10 * GENE_SIZE + sum(gene**2 - 10 * cos(2 * pi * gene) for gene in individual.gene)


def second_minimisation():
    for individual in population:
        firstSum = 0.0
        secondSum = 0.0
        for c in individual.gene:
            firstSum += c ** 2.0
            secondSum += cos(2.0 * pi * c)
        n = float(GENE_SIZE)
        individual.fitness = -20.0 * exp(-0.2 * sqrt(firstSum / n)) - exp(secondSum / n) + 20 + e


def get_sorted_population_list():
    return sorted(population, key=lambda indi: indi.fitness)


def sort_population():
    sorted_population = [population[0]]
    for individual in population[1:]:
        i = 0
        for sorted_individual in sorted_population:
            if MINIMISATION:
                if individual.fitness < sorted_individual.fitness:
                    sorted_population.insert(i, individual)
                    break
            else:
                if individual.fitness > sorted_individual.fitness:
                    sorted_population.insert(i, individual)
                    break
            i += 1
        if i == len(sorted_population):
            sorted_population.append(individual)
    return sorted_population


def population_print():
    sorted_pop = get_sorted_population_list()
    print("#################")
    for print_individual in sorted_pop:
        print("Fitness:", print_individual.fitness, print_individual.gene)
    print("#################")


def save_data(run, generation):
    data = open(f"{DATA_PATH}\\{run}.csv", "a")
    fitness = 0
    sorted_population = sort_population()
    for individual in population:
        fitness += individual.fitness
    fitness = fitness / POP_SIZE
    data.write(f"{sorted_population[0].fitness},{fitness},{sorted_population[POP_SIZE - 1].fitness}\n")
    if generation == 0 or generation + 1 == NUM_GENERATIONS or generation == int(NUM_GENERATIONS/2):
        print(f"{sorted_population[0].fitness},{fitness},{sorted_population[POP_SIZE - 1].fitness}\n")
    data.close()


def save_final_pop():
    final_pop = open(f"{DATA_PATH}\\final_pop.csv", "a")
    sorted_pop = sort_population()
    for individual in population:
        for gene in individual.gene:
            final_pop.write(f"{gene},")
        final_pop.write(f"{individual.fitness}\n")
    final_pop.close()


def chance_of(chance):
    if random() * 100 > chance:
        return False
    return True


def trunc_gauss(mu, sigma, bottom, top):
    a = gauss(mu,sigma)
    while (bottom <= a <= top) == False:
        a = gauss(mu, sigma)
    return a


if __name__ == '__main__':
    if DEBUG:
        seed(1000)
    for run in range(0, MAX_RUNS):
        print("Run", run)
        file = open(f"{DATA_PATH}\\{run}.csv", "w+")
        file.close()
        main(run)
        if PRINT_FINAL_POPULATION:
            save_final_pop()

