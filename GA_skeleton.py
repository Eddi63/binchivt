# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""
import timeit

import numpy as np


def mutation(a, local_state):
    """ swap two random indexes """
    index_1 = local_state.randint(0, len(a))
    index_2 = local_state.randint(0, len(a))
    a[index_1], a[index_2] = a[index_2], a[index_1]
    return a


def select_proportional(Genome, fitness, rand_state):
    ''' RWS: Select one individual out of a population Genome with fitness values fitness using proportional selection.'''
    cumsum_f = np.cumsum(fitness)
    r = sum(fitness) * rand_state.uniform()
    idx = np.ravel(np.where(r < cumsum_f))[0]

    return Genome[idx]


def two_point_crossover(a, b, local_state):
    index_1 = local_state.randint(0, len(a))
    index_2 = local_state.randint(0, len(b))
    if index_1 > index_2:
        index_1, index_2 = index_2, index_1
    child_1 = np.copy(a[index_1:index_2])  # copying section for cross
    child_2 = np.copy(b[index_1:index_2])
    child_1 = np.concatenate([child_1, [i for i in b if i not in child_1]])  # bookkeeping for legal permuation
    child_2 = np.concatenate([child_2, [i for i in a if i not in child_2]])
    return child_1, child_2


def some_crossover(graph, p1, p2, local_state, r=0): #generating one child from best recombination seqentially
    index_1 = local_state.randint(0, len(p1)) #Sequential Constructive Crossover Operator." (Zakir H. Ahmed)
    index_2 = local_state.randint(0, len(p2))
    if index_1 > index_2:
        index_1, index_2 = index_2, index_1
    if r%2 :
        parent = p1
    else:
        parent = p2
    c1 = np.zeros(150)  # copying section for cross
    c1[0:index_2-index_1] = np.copy ( parent[index_1:index_2])

    for i in range ((index_2 - index_1 - 1 ), len(p1) - 1 ):
        for idx1 in range(len(p1)):
            if p1[idx1] in c1 :
                idx1_can = (idx1 + 1) % len(p1)
            else:
                idx1_can = idx1
                break

        for idx2 in range(len(p2)):
            if p2[idx2] in c1:
                idx2_can = (idx2 + 1) % len(p2)
            else:
                idx2_can = idx2
                break

        dist1 = graph[int(c1[i]), int(p1[idx1_can])]
        dist2 = graph[int(c1[i]), int(p2[idx2_can])]
        if dist1 < dist2:
            c1[i+1] = p1[idx1_can]
        else:
            c1[i + 1] = p2[idx2_can]
    return c1



def GA(n, max_evals, fitnessfct, graph, seed=None, selectfct=select_proportional):
    eval_cntr = 0
    history = []
    fmin = np.inf
    xmin = np.array([n, 1])
    t1 = timeit.default_timer()
    mu = 100
    pc = 0.37
    pm = 2 / n
    local_state = np.random.RandomState(seed)

    Genome = np.array([local_state.permutation(n) for _ in range(mu)])  # choice of alphabet : ints 1-150
    fitnessPop = []
    for i in range(mu):
        fitnessPop.append(fitnessfct(Genome[i], graph))

    eval_cntr += mu
    fcurr_best = fmin = np.min(fitnessPop)
    xmin = Genome[np.argmin(fitnessPop)]
    history.append(fmin)

    while (eval_cntr < max_evals):
        newGenome = np.empty([mu, n], dtype=int)
        for i in range(int(mu / 2)):
            p1 = selectfct(Genome, fitnessPop, local_state)
            p2 = selectfct(Genome, fitnessPop, local_state)
            if local_state.uniform() < pc:

                c1 = some_crossover(graph, p1, p2, local_state,0)
                c2 = some_crossover(graph, p1, p2, local_state,1)

                # c1, c2 = two_point_crossover(p1,p2, local_state) #crossover
                # c1, c2 = max_edge_crossover(graph, p1, p2, local_state)  # crossover
            else:
                c1, c2 = np.copy(p1), np.copy(p2)  # elitism

            if local_state.uniform() < pm:  # mutation of childs
                c1 = mutation(c1, local_state)
            if local_state.uniform() < pm:
                c2 = mutation(c2, local_state)

            newGenome[2 * i - 1] = np.copy(c1)
            newGenome[2 * i] = np.copy(c2)

        newGenome[mu - 1] = np.copy(Genome[np.argmin(fitnessPop)])
        Genome = np.copy(newGenome)

        fitnessPop.clear()
        for i in range(mu):
            fitnessPop.append(fitnessfct(Genome[i], graph))

        eval_cntr += mu
        fmin = np.min(fitnessPop)

        xmin = Genome[:, [np.argmin(fitnessPop)]]
        history.append(fmin)
        if fmin < fcurr_best:
            fcurr_best = fmin
            xmin = Genome[:, [np.argmin(fitnessPop)]]

        history.append(fcurr_best)
        if np.mod(eval_cntr, int(max_evals / 100)) == 0:
            t2 = timeit.default_timer()
            print(eval_cntr, " evals: fmin=", fmin)
            #print(eval_cntr, " evals: xmin=", xmin)
            print("time for 10**3 iterations:", t2-t1)
        local_state = np.random.RandomState(seed + eval_cntr)
        if fmin < 6300:
            print(eval_cntr, " evals: fmin=", fmin, "; done!")
            #print(eval_cntr, " evals: xmin=", xmin, "; done!")
            break

    return xmin, fmin, history
