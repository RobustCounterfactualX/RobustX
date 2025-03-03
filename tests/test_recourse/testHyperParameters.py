import itertools
import random

import rocoursenet_test

33

def main():
    max_iter = [60]
    lr = [0.002, 0.003]
    delta = [0.05]
    n_attacker_steps = [50,60]
    lambda_ = [0.5]

    params = list(itertools.product(max_iter, lr, delta, n_attacker_steps, lambda_))
    done = {}
    count = 0
    while count != len(params):
        x = random.choice(params)
        while x in done:
            x = random.choice(params)
        try:
            print(f'trying {x}')
            rocoursenet_test.test_rocoursenet(*x)
            print(x)
            break
        except AssertionError as e:
            print('done')
            pass

        print(count)
        count+=1






main()