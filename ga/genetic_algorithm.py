from track_generator.utils import *
from euclid import Point2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class GeneticAlgorithm:

    def __init__(self):
        self.top_selection_n = int(TOP_SELECTION_RATIO * POPULATION_SIZE)
        self.population, self.population_fitness = self._initialization()
        self.generation = 0

    def _initialization(self):
        """Randomly generates the initial population (generation 0)"""
        print('* Generating initial population (generation 0)')
        population, population_fitness = [], []
        for t_no in range(POPULATION_SIZE):
            # Get genotypes of random track, convert to phenotypes and get track points
            instructions, rotations = get_random_track()
            instructions_sequence = genotype_to_phenotype(instructions, rotations)
            track_points = generate_track(chromosome_elements=instructions_sequence)
            fitness = self._calc_fitness(track_points)

            population.append([instructions, rotations])
            population_fitness.append(fitness)

        population = np.array(population)
        population_fitness = np.array(population_fitness)

        return population, population_fitness

    def _selection(self):
        """Selects the top individuals in the population to create the mating pool"""
        selected_idx = self.population_fitness.argsort()[:self.top_selection_n]
        self.mating_pool = self.population[selected_idx]

    @staticmethod
    def _crossover(parent_1, parent_2, crossover_pt):
        """Creates offspring by combining genotypes of both the parents acc to the crossover point"""
        instructions = np.concatenate([parent_1[0][:crossover_pt], parent_2[0][crossover_pt:]]).astype(int)
        rotations = np.concatenate([parent_1[1][:crossover_pt], parent_2[1][crossover_pt:]])

        return [list(instructions), list(rotations)]

    @staticmethod
    def _mutate(instructions, rotations):
        """Performs mutation by modifying part of the track randomly"""

        # Randomly determine the length of the track to be modified and the start point of mutation
        mutate_length = np.random.randint(low=2, high=1 + (len(instructions) // 3))
        mutation_start = np.random.randint(low=1, high=len(instructions) - mutate_length)

        # Extract the part of track before mutated portion
        instructions_start = instructions[:mutation_start]
        rotations_start = rotations[:mutation_start]

        mutated_inst, mutated_rot = get_random_track(
            instructions_start,
            rotations_start,
            mutate_length + len(instructions_start))

        # Add the part of the track after mutated portion
        mutated_inst += instructions[mutation_start+mutate_length:]
        mutated_rot += rotations[mutation_start+mutate_length:]

        return mutated_inst, mutated_rot

    def _reproduction(self):
        """Generates the next generation using crossover and mutation of parents in the mating pool"""

        new_population, new_population_fitness = [], []
        new_pop_size = 0
        # randomly determine the crossover point for each offspring
        crossover_points = np.random.randint(low=1, high=CHROMOSOME_LENGTH, size=POPULATION_SIZE)

        while new_pop_size < POPULATION_SIZE:

            # randomly get parents from the mating pool
            parent_1_idx, parent_2_idx = np.random.choice(range(len(self.mating_pool)), size=2, replace=False)

            # perform crossover to produce offspring
            instructions, rotations = self._crossover(self.mating_pool[parent_1_idx], self.mating_pool[parent_2_idx],
                                                      crossover_points[new_pop_size])

            # Perform mutation
            if np.random.random() < MUTATION_PROB:
                instructions, rotations = self._mutate(instructions, rotations)

            instructions_sequence = genotype_to_phenotype(instructions, rotations)
            track_points = generate_track(chromosome_elements=instructions_sequence)

            # If offspring contains loop, reject it
            if (has_loop(track_points)) | (len(instructions) != CHROMOSOME_LENGTH):
                continue
            else:
                fitness = self._calc_fitness(track_points)

                new_population.append([instructions, rotations])
                new_population_fitness.append(fitness)
                new_pop_size += 1

        self.population = np.array(new_population)
        self.population_fitness = np.array(new_population_fitness)

    def _calc_fitness(self, track_points):
        """Calculates the fitness value - distance between the start end end points of the track"""
        start_pt = Point2(track_points[0].x, track_points[0].y)
        end_pt = Point2(track_points[-1].x, track_points[-1].y)

        return start_pt.distance(end_pt)

    def display_random_samples(self):
        """Randomly displays 6 samples from the current population"""
        cols = 3
        rows = 2

        random_idx = np.random.choice(range(len(self.population)), size=cols * rows, replace=False)
        random_individuals = self.population[random_idx]

        fig = plt.figure(figsize=(12, 8))
        plt.suptitle(f'Generation {self.generation} samples')
        for i in range(cols * rows):
            instructions, rotations = random_individuals[i]
            instructions_sequence = genotype_to_phenotype(instructions.astype(int), rotations)
            track_points = generate_track(chromosome_elements=instructions_sequence)

            plot_x = [track_point.x for track_point in track_points]
            plot_y = [track_point.y for track_point in track_points]
            fig.add_subplot(rows, cols, i + 1)
            plt.scatter(plot_x, plot_y)
            plt.plot(plot_x[0], plot_y[0], 'go', markersize=10, label='Start point')
            plt.plot(plot_x[-1], plot_y[-1], 'ko', markersize=10, label='End point')
            plt.xticks([])
            plt.yticks([])
            instructions_seq_text = ', '.join([f'({i.command.name} {round(i.value, 2)})' for i in instructions_sequence])
            fitness = self._calc_fitness(track_points)

            plt.title(f'[{instructions_seq_text}]\nFitness = {fitness:.4f}', fontsize=8)
        plt.tight_layout()
        # plt.show()

    def display_fittest(self):
        fittest_idx = self.population_fitness.argsort()[0]

        instructions, rotations = self.population[fittest_idx]
        instructions_sequence = genotype_to_phenotype(instructions.astype(int), rotations)
        track_points = generate_track(chromosome_elements=instructions_sequence)

        plot_x = [track_point.x for track_point in track_points]
        plot_y = [track_point.y for track_point in track_points]
        plt.scatter(plot_x, plot_y)
        plt.plot(plot_x[0], plot_y[0], 'go', markersize=10, label='Start point')
        plt.plot(plot_x[-1], plot_y[-1], 'ko', markersize=10, label='End point')
        plt.xticks([])
        plt.yticks([])
        instructions_seq_text = ', '.join([f'({i.command.name} {round(i.value, 2)})' for i in instructions_sequence])
        fitness = self._calc_fitness(track_points)

        plt.title(f'[{instructions_seq_text}]\nFitness = {fitness:.4f}', fontsize=8)
        plt.show()

    def run(self, display_samples=False):

        mean_fitness = self.population_fitness.mean()
        min_fitness = self.population_fitness.min()
        max_fitness = self.population_fitness.max()
        fitness_gen = [[min_fitness, mean_fitness, max_fitness]]
        print(f'--> Generation {self.generation}:')
        print(f'Mean fitness = {mean_fitness:.4f}, '
              f'Min fitness = {min_fitness:.4f}, '
              f'Max fitness = {max_fitness:.4f}')
        if display_samples:
            self.display_random_samples()
        for gen in range(NUM_GENERATIONS):
            self.generation = gen + 1
            print(f'\n--> Generation {self.generation}:')
            print('* Selection')
            self._selection()

            print('* Reproduction')
            self._reproduction()

            mean_fitness = self.population_fitness.mean()
            min_fitness = self.population_fitness.min()
            max_fitness = self.population_fitness.max()
            fitness_gen.append([min_fitness, mean_fitness, max_fitness])
            print(f'Mean fitness = {mean_fitness:.4f}, '
                  f'Min fitness = {min_fitness:.4f}, '
                  f'Max fitness = {max_fitness:.4f}')

            if display_samples:
                self.display_random_samples()

        # Plot evolution of fitness across generations
        plt.figure(figsize=(8, 5))
        plt.plot(np.array(fitness_gen))
        plt.legend(['Min fitness', 'Mean fitness', 'Max fitness'])
        plt.xlabel('Generation', fontsize=13)
        plt.ylabel('Fitness', fontsize=13)
        plt.show()

        # Plot track with minimum fitness
        self.display_fittest()
