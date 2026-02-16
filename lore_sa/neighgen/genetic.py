import numbers
import pickle
import random

from deap.algorithms import varAnd, eaSimple
from scipy.spatial.distance import cdist, hamming
from .neighborhood_generator import NeighborhoodGenerator
from deap import base, creator, tools, algorithms

import numpy as np

__all__ = ["NeighborhoodGenerator", "GeneticGenerator"]

from ..util import neuclidean, sigmoid


class LegacyGeneticGenerator(NeighborhoodGenerator):
    """
    Random Generator creates neighbor instances by generating random values starting from an input instance and
    pruning the generation around a fitness function based on proximity to the instance to explain
    """
    def __init__(self, bbox=None, dataset=None, encoder=None, ocr=0.1,
                 alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, random_seed=None):
        """

        :param bbox: the Black Box model to explain
        :param dataset: the dataset with the descriptor of the original dataset
        :param encoder: an encoder to transfrom the data from/to the black box model
        :param ocr: acronym for One Class Ratio, it is the ratio of the number of instances of the minority class
        :param alpha1: the weight of the similarity of the features from the given instance. The sum of alpha1 and alpha2 must be 1
        :param alpha2: the weight of the similiarity of the target class from the given instance. The sum of alpha1 and alpha2 must be 1
        :param metric: the distance metric to use to compute the distance between instances
        :param ngen: the number of generations to run
        :param mutpb: probability of mutation of a specific feature
        :param cxpb:
        :param tournsize:
        :param halloffame_ratio:
        :param random_seed: initial seed for the random number generator
        """
        self.bbox = bbox
        self.dataset = dataset
        self.encoder = encoder
        self.ocr = ocr
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.metric = metric
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.tournsize = tournsize
        self.halloffame_ratio = halloffame_ratio
        self.random_seed = random_seed
        random.seed(random_seed)

    def generate(self, z, num_instances, descriptor, encoder):
        """
        The generation is based on the strategy of generating a number of instances for the same class as the input
        instance and a number of instances for a different class.
        The generation of the instances for each subgroup is done through a genetic algorithm based on two fitness
        fuctions: one for the same class and one for the different class.
        :param z: the input instance
        :param num_instances: how many elements to generate
        :param descriptor: the descriptor of the dataset
        :return:
        """
        new_x = z.copy()

        # determine the number of instances to generate for the same class and for a different class
        num_samples_eq = int(np.round(num_instances * 0.5))
        num_samples_neq = num_instances - num_samples_eq

        # generate the instances for the same class
        toolbox_eq = self.setup_toolbox(z, self.fitness_equal, num_samples_eq)
        population_eq, halloffame_eq, logbook_eq = self.fit(toolbox_eq, num_samples_eq)
        Z_eq = self.add_halloffame(population_eq, halloffame_eq)
        # print(logbook_eq)

        # generate the instances for a different class
        toolbox_noteq = self.setup_toolbox(z, self.fitness_notequal, num_samples_neq)
        population_noteq, halloffame_noteq, logbook_noteq = self.fit(toolbox_noteq, num_samples_neq)
        Z_noteq = self.add_halloffame(population_noteq, halloffame_noteq)
        # print(logbook_noteq)

        # concatenate the two sets of instances
        Z = np.concatenate((Z_eq, Z_noteq), axis=0)

        # balance the instances according to the minority class
        Z = super(LegacyGeneticGenerator, self).balance_neigh(z, Z, num_instances)
        # the first element is the input instance

        Z[0] = new_x
        return Z

    def add_halloffame(self, population, halloffame):
        fitness_values = [p.fitness.wvalues[0] for p in population]
        fitness_values = sorted(fitness_values)
        fitness_diff = [fitness_values[i + 1] - fitness_values[i] for i in range(0, len(fitness_values) - 1)]

        sorted_array = np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist()
        if len(sorted_array) == 0:
            fitness_value_thr = -np.inf
        else:
            index = np.max(sorted_array)
            fitness_value_thr = fitness_values[index]

        Z = list()
        for p in population:
            # if p.fitness.wvalues[0] > fitness_value_thr:
            Z.append(p)

        for h in halloffame:
            if h.fitness.wvalues[0] > fitness_value_thr:
                Z.append(h)

        return np.array(Z)

    def setup_toolbox(self, x, evaluate, population_size):
        # if hasattr(creator, "fitness"):
        #     del creator.fitness
        # if hasattr(creator, "individual"):
        #     del creator.individual

        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", np.ndarray, fitness=creator.fitness)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x)
        toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)

        toolbox.register("clone", self.clone)
        # toolbox.register("evaluate", self.constraint_decorator(evaluate, x))
        toolbox.register("evaluate", evaluate, x)
        toolbox.register("mate", self.mate)
        toolbox.register("mutate", self.mutate, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

        return toolbox

    def setup_toolbox_noteq(self, x, x1, evaluate, population_size):

        creator.create("fitness_noteg", base.Fitness, weights=(1.0,))
        creator.create("individual_noteq", np.ndarray, fitness=creator.fitness_noteq)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x1)
        toolbox.register("individual", tools.initIterate, creator.individual_noteq, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual_noteq, n=population_size)

        toolbox.register("clone", self.clone)
        toolbox.register("evaluate", evaluate, x)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

        return toolbox

    def fit(self, toolbox, population_size):

        halloffame_size = int(np.round(population_size * self.halloffame_ratio))

        population = toolbox.population(n=population_size)
        halloffame = tools.HallOfFame(halloffame_size, similar=np.array_equal)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        population, logbook = eaSimple(population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                                  ngen=self.ngen, stats=stats, halloffame=halloffame,
                                                  verbose=True)

        return population, halloffame, logbook

    def record_init(self, x):
        '''
        This function is used to generate a random instance to start the evolutionary algorithm.
        In this case we repeat the input instance x for all the initial population

        :return: a (not so) random instance
        '''
        return x

    def random_init(self):
        z = self.generate_synthetic_instance()

        return z

    def clone(self, x):
        return pickle.loads(pickle.dumps(x))

    def mutate(self, toolbox, x):
        z = toolbox.clone(x)
        nz = self.generate_synthetic_instance(from_z=z, mutpb=self.mutpb)

        z[:] = nz

        return z,

    def mate(self, ind1, ind2):
        """Executes a two-point crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.
        This implementation uses the original implementation of the DEAP library. It adds a special case for the
        one-hot encoding, where the crossover is done taking into account the intervals of values imposed by
        the one-hot encoding.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.randint` function from the Python
        base :mod:`random` module.
        """
        if self.encoder.type == 'one-hot':
            intervals = self.encoder.get_encoded_intervals()
            cxInterval1 = random.randint(0, len(intervals) - 1)
            cxInterval2 = random.randint(0, len(intervals) - 1)
            if cxInterval1 > cxInterval2:
                # Swap the two cx intervals
                cxInterval1, cxInterval2 = cxInterval2, cxInterval1

            cxpoint1 = intervals[cxInterval1][0]
            cxpoint2 = intervals[cxInterval2][1]
            ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
                = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        else:
            size = min(len(ind1), len(ind2))
            cxpoint1 = random.randint(1, size)
            cxpoint2 = random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:  # Swap the two cx points
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
                = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

        return ind1, ind2



    def fitness_equal(self, z, z1):
        if isinstance(self.metric, numbers.Number):
            self.metric = neuclidean
        feature_similarity_score = 1.0 - cdist(z.reshape(1, -1), z1.reshape(1, -1), metric=self.metric).ravel()[0]
        # feature_similarity = feature_similarity_score if feature_similarity_score >= self.eta1 else 0.0
        feature_similarity = sigmoid(feature_similarity_score) if feature_similarity_score < 1.0 else 0.0
        # feature_similarity = sigmoid(feature_similarity_score)

        # y = self.bb_predict(x.reshape(1, -1))[0]
        # y1 = self.bb_predict(x1.reshape(1, -1))[0]
        x = self.encoder.decode(z.reshape(1, -1))
        y = self.bbox.predict(x)

        x1 = self.encoder.decode(z1.reshape(1, -1))
        # if None in x1[0]:
        #     x1 = x
        y1 = self.bbox.predict(x1)

        target_similarity_score = 1.0 - hamming(y, y1)
        # target_similarity = target_similarity_score if target_similarity_score >= self.eta2 else 0.0
        target_similarity = sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,


    def fitness_notequal(self, z, z1):
        feature_similarity_score = 1.0 - cdist(z.reshape(1, -1), z1.reshape(1, -1), metric=self.metric).ravel()[0]
        # feature_similarity = feature_similarity_score if feature_similarity_score >= self.eta1 else 0.0
        feature_similarity = sigmoid(feature_similarity_score)

        # y = self.bb_predict(x.reshape(1, -1))[0]
        # y1 = self.bb_predict(x1.reshape(1, -1))[0]
        x = self.encoder.decode(z.reshape(1, -1))
        y = self.bbox.predict(x)

        x1 = self.encoder.decode(z1.reshape(1, -1))
        # if None in x1[0]:
        #     x1 = x
        y1 = self.bbox.predict(x1)

        target_similarity_score = 1.0 - hamming(y, y1)
        # target_similarity = target_similarity_score if target_similarity_score < self.eta2 else 0.0
        target_similarity = 1.0 - sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,



class GeneticGenerator(LegacyGeneticGenerator):
    """
    Random Generator creates neighbor instances by generating random values starting from an input instance and
    pruning the generation around a fitness function based on proximity to the instance to explain
    """
    def __init__(self, bbox=None, dataset=None, encoder=None, ocr=0.1,
                 alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=30, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, random_seed=None):
        """

        :param bbox: the Black Box model to explain
        :param dataset: the dataset with the descriptor of the original dataset
        :param encoder: an encoder to transfrom the data from/to the black box model
        :param ocr: acronym for One Class Ratio, it is the ratio of the number of instances of the minority class
        :param alpha1: the weight of the similarity of the features from the given instance. The sum of alpha1 and alpha2 must be 1
        :param alpha2: the weight of the similiarity of the target class from the given instance. The sum of alpha1 and alpha2 must be 1
        :param metric: the distance metric to use to compute the distance between instances
        :param ngen: the number of generations to run
        :param mutpb: probability of mutation of a specific feature
        :param cxpb:
        :param tournsize:
        :param halloffame_ratio:
        :param random_seed: initial seed for the random number generator
        """
        self.bbox = bbox
        self.dataset = dataset
        self.encoder = encoder
        self.ocr = ocr
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.metric = metric
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.tournsize = tournsize
        self.halloffame_ratio = halloffame_ratio
        self.random_seed = random_seed
        random.seed(random_seed)

    def generate(self, z, num_instances, descriptor, encoder):
        """
        The generation is based on the strategy of generating a number of instances for the same class
        as the input instance and a number of instances for a different class.
        The generation of the instances for each subgroup is done through a genetic algorithm based on
        two fitness fuctions: one for the same class and one for the different class.
        :param z: the input instance, from which the generation starts
        :param num_instances: how many elements to generate
        :param descriptor: the descriptor of the dataset. This provides the metadata of each feature to guide the generation
        :param encoder: the encoder to transform the data from/to the black box model

        :return: a new set of instances generated from the input instance. The first element is the input instance
        """
        new_x = z.copy()

        # determine the number of instances to generate for the same class and for a different class
        num_samples_eq = int(np.round(num_instances * 0.5))
        num_samples_neq = num_instances - num_samples_eq

        # generate the instances for the same class
        toolbox_eq = self.setup_toolbox(z, self.population_fitness_equal(z), num_samples_eq)
        population_eq, halloffame_eq, logbook_eq = self.fit(toolbox_eq, num_samples_eq)
        Z_eq = self.add_halloffame(population_eq, halloffame_eq)
        # print(logbook_eq)

        # generate the instances for a different class
        toolbox_noteq = self.setup_toolbox(z, self.population_fitness_notequal(z), num_samples_neq)
        population_noteq, halloffame_noteq, logbook_noteq = self.fit(toolbox_noteq, num_samples_neq)
        Z_noteq = self.add_halloffame(population_noteq, halloffame_noteq)
        # print(logbook_noteq)

        # concatenate the two sets of instances
        Z = np.concatenate((Z_eq, Z_noteq), axis=0)

        # balance the instances according to the minority class
        Z = super(GeneticGenerator, self).balance_neigh(z, Z, num_instances)
        # the first element is the input instance

        Z[0] = new_x
        return Z

    # def add_halloffame(self, population, halloffame):
    #     fitness_values = [p.fitness.wvalues[0] for p in population]
    #     fitness_values = sorted(fitness_values)
    #     fitness_diff = [fitness_values[i + 1] - fitness_values[i] for i in range(0, len(fitness_values) - 1)]
    #
    #     sorted_array = np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist()
    #     if len(sorted_array) == 0:
    #         fitness_value_thr = -np.inf
    #     else:
    #         index = np.max(sorted_array)
    #         fitness_value_thr = fitness_values[index]
    #
    #     Z = list()
    #     for p in population:
    #         # if p.fitness.wvalues[0] > fitness_value_thr:
    #         Z.append(p)
    #
    #     for h in halloffame:
    #         if h.fitness.wvalues[0] > fitness_value_thr:
    #             Z.append(h)
    #
    #     return np.array(Z)

    def setup_toolbox(self, x, evaluate, population_size):
        # if hasattr(creator, "fitness"):
        #     del creator.fitness
        # if hasattr(creator, "individual"):
        #     del creator.individual

        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", np.ndarray, fitness=creator.fitness)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x)
        toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)

        toolbox.register("clone", self.clone)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", self.mate)
        toolbox.register("mutate", self.mutate, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

        return toolbox


    def fit(self, toolbox, population_size):

        halloffame_size = int(np.round(population_size * self.halloffame_ratio))

        population = toolbox.population(n=population_size)
        halloffame = tools.HallOfFame(halloffame_size, similar=np.array_equal)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)

        population, logbook = GeneticGenerator.eaSimple(population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                                  ngen=self.ngen, stats=stats, halloffame=halloffame,
                                                  verbose=False)

        return population, halloffame, logbook

    @staticmethod
    def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
                 halloffame=None, verbose=__debug__):
        """This algorithm reproduce the simplest evolutionary algorithm as
        presented in chapter 7 of [Back2000]_.

        :param population: A list of individuals.
        :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                        operators.
        :param cxpb: The probability of mating two individuals.
        :param mutpb: The probability of mutating an individual.
        :param ngen: The number of generation.
        :param stats: A :class:`~deap.tools.Statistics` object that is updated
                      inplace, optional.
        :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                           contain the best individuals, optional.
        :param verbose: Whether or not to log the statistics.
        :returns: The final population
        :returns: A class:`~deap.tools.Logbook` with the statistics of the
                  evolution

        This implementation is an adaptation of the original algorithm implemented in the DEAP library.

        .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
           Basic Algorithms and Operators", 2000.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.evaluate(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit, )

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook


    def population_fitness_equal(self, z):
        """
        This fitness function evaluate the feature_similarity and the target_similarity of a population against a given
        instance z. The two similarities are computed using optimezed functions of `numpy` and `scipy` libraries.
        This improves the performance of the algorithm.
        """
        def wrapper(population):
            if isinstance(self.metric, numbers.Number):
                self.metric = neuclidean
            feature_similarity_score = 1.0 - cdist(z.reshape(1, -1), population, metric=self.metric).ravel()
            feature_similarity = sigmoid(feature_similarity_score)

            x = self.encoder.decode(z.reshape(1, -1))
            pop = self.encoder.decode(np.array(population))
            pop_y = self.bbox.predict(pop)
            y = self.bbox.predict(x)

            target_similarity = np.array([sigmoid(1.0 - hamming(y, [y1])) for y1 in pop_y])


            evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity

            return evaluation
        return wrapper

    def population_fitness_notequal(self, z):
        def wrapper(population):
            if isinstance(self.metric, numbers.Number):
                self.metric = neuclidean
            feature_similarity_score = 1.0 - cdist(z.reshape(1, -1), population, metric=self.metric).ravel()
            feature_similarity = sigmoid(feature_similarity_score)

            x = self.encoder.decode(z.reshape(1, -1))
            pop = self.encoder.decode(np.array(population))
            pop_y = self.bbox.predict(pop)
            y = self.bbox.predict(x)

            target_similarity = np.array([sigmoid(hamming(y, [y1])) for y1 in pop_y])


            evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity

            return evaluation
        return wrapper