import copy
import random
import numpy as np


def check_max_prog_size(MAX_PROGRAM_SIZE):
    pass


def two_point_crossover(prog1, prog2, MAX_PROGRAM_SIZE):
    prog1_matrix = np.array(prog1)
    prog2_matrix = np.array(prog2)

    min_size = min(len(prog1), len(prog2))

    xover_p1 = random.randint(1, min_size)
    xover_p2 = random.randint(1, min_size - 1)

    if xover_p2 >= xover_p1:
        xover_p2 += 1
    else:
        # Swapping the two xover points
        xover_p1, xover_p2 = xover_p2, xover_p1

    org_prog1_matrix = copy.deepcopy(prog1_matrix)
    prog1_matrix[xover_p1:xover_p2] = prog2_matrix[xover_p1:xover_p2]
    prog2_matrix[xover_p1:xover_p2] = org_prog1_matrix[xover_p1:xover_p2]

    return prog1_matrix.tolist(), prog2_matrix.tolist()


def crossover(selected_programs, gap, MAX_PROGRAM_SIZE):
    offsprings = []

    for i in range(int(gap/2)):
        offspring1, offspring2 = two_point_crossover(
            selected_programs[i], selected_programs[gap-1-i], MAX_PROGRAM_SIZE)

        offsprings.append(offspring1)
        offsprings.append(offspring2)

    return offsprings


def swap_mutation(offspring):
    size = range(len(offspring))
    i, j = random.sample(size, 2)
    # Swapping
    old_i_element = offspring[i]
    offspring[i] = offspring[j]
    offspring[j] = old_i_element

    return offspring


def mutFlipBit(individual, indpb):
    """Flip the value of the attributes of the input individual and return the
    mutant. The *individual* is expected to be a :term:`sequence` and the values of the
    attributes shall stay valid after the ``not`` operator is called on them.
    The *indpb* argument is the probability of each attribute to be
    flipped. This mutation is usually applied on boolean individuals.
    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be flipped.
    :returns: A tuple of one individual.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i in xrange(len(individual)):
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])

    return individual


def mutation(offsprings):
    mutated_offsprings = []

    for offspring in offsprings:
        mutated_offspring = swap_mutation(offspring)
        mutated_offsprings.append(mutated_offspring)

    return mutated_offsprings
