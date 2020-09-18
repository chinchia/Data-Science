import numpy as np

# you must use python 3.6, 3.7, 3.8(3.8 not for macOS) for sourcedefender
import sourcedefender
from HomeworkFramework import Function

import datetime
start_time = datetime.datetime.now()

# Particle Swarm Optimization
def PSO(main_info, max_iters=50, pop_size=10, c1=1.5, c2=1.5, w=0.7, w_step=1.0):
    np.random.seed(5)
    
    cost_func = main_info['cost_function']
    lower_bounds = main_info['lower_bounds']
    upper_bounds = main_info['upper_bounds']
    n_dim = main_info['n_dim']

    # empty particle template
    empty_particle = {'position': None,
                      'velocity': None,
                      'cost': None,
                      'best_position': None,
                      'best_cost': None,
                     }

    # initialize global best
    gbest = {'position': None,
             'cost': np.inf}

    # create initial population
    population = []
    for i in range(pop_size):
        population.append(empty_particle.copy())
        population[i]['position'] = np.random.uniform(lower_bounds, upper_bounds, n_dim)
        population[i]['velocity'] = np.zeros(n_dim)
        population[i]['cost'] = cost_func(population[i]['position'])
        population[i]['best_position'] = population[i]['position'].copy()
        population[i]['best_cost'] = population[i]['cost']
        
        if population[i]['best_cost'] < gbest['cost']:
            gbest['position'] = population[i]['best_position'].copy()
            gbest['cost'] = population[i]['best_cost']
    
    # update gbest
    for it in range(max_iters):
        for i in range(pop_size):
            population[i]['velocity'] = w * population[i]['velocity'] \
                + c1 * np.random.rand(n_dim) * (population[i]['best_position'] - population[i]['position']) \
                + c2 * np.random.rand(n_dim) * (gbest['position'] - population[i]['position'])

            population[i]['position'] += population[i]['velocity']
            population[i]['position'] = np.maximum(population[i]['position'], lower_bounds)
            population[i]['position'] = np.minimum(population[i]['position'], upper_bounds)
            population[i]['cost'] = cost_func(population[i]['position'])
            
            if population[i]['cost'] < population[i]['best_cost']:
                population[i]['best_position'] = population[i]['position'].copy()
                population[i]['best_cost'] = population[i]['cost']

                if population[i]['best_cost'] < gbest['cost']:
                    gbest['position'] = population[i]['best_position'].copy()
                    gbest['cost'] = population[i]['best_cost']

        w *= w_step

    return gbest, population

class PSO_optimizer(Function): # need to inherit this class "Function"
    
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)        

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value
    
    def fitness(self, x):
        return self.f.evaluate(func_num, x)

    def run(self, FES, params_list): # main part for your implementation
        
        #while self.eval_times < FES:
        #print('=====================FE=====================')
        #print(self.eval_times)
            
        main_info = {'cost_function': self.fitness,
                     'lower_bounds': self.lower,
                     'upper_bounds': self.upper,
                     'n_dim': self.dim,
                    }
        
        c1, c2, w, w_step = params_list[0], params_list[1], params_list[2], params_list[3]
        max_iters = int(80 * 0.3 * func_num)
        pop_size = int(12 + 2 * func_num)
        
        gbest, _ = PSO(main_info, max_iters, pop_size, c1, c2, w, w_step)
            
        solution = gbest['position']
        value = gbest['cost']

        #if value == "ReachFunctionLimit":
            #print("ReachFunctionLimit")
            #break
            
        if float(value) < self.optimal_value:
            self.optimal_solution[:] = solution
            self.optimal_value = float(value)

        #print("optimal: %f\n" % self.get_optimal()[1])            

if __name__ == '__main__':
    
    try_times = 1
    best_func = {}
    min_func1, min_func2, min_func3, min_func4 = 9999, 9999, 9999, 9999
    
    random_params_list = []
    for i in range(2500):
        random_params_list.append([np.random.uniform(1,2), np.random.uniform(1,2), 
                                   np.random.uniform(0,1), np.random.uniform(0.8,1)])
        
    params = random_params_list + [[1.3215948000000002,1.7231869000000002,0.6069247999999998,0.9064781999999997],
                                   [1.3215951000000004,1.7231871000000003,0.6069252999999996,0.9064781999999997],
                                   [1.3215951000000004,1.7231872000000004,0.6069254999999995,0.9064782999999996]]
    
    for param in params:
        #print('\n[try ' + str(try_times) + ']\n')
        
        func_num = 1
        fes = 0
        #function1: 1000, function2: 1500, function3: 2000, function4: 2500
        while func_num < 5:
            if func_num == 1:
                fes = 1000
            elif func_num == 2:
                fes = 1500
            elif func_num == 3:
                fes = 2000 
            else:
                fes = 2500

            # you should implement your optimizer
            op = PSO_optimizer(func_num)
            op.run(fes, param)

            best_input, best_value = op.get_optimal()
            
            if func_num == 1 and best_value < min_func1:
                best_func['1'] = [best_input, best_value]
                min_func1 = best_value
            elif func_num == 2 and best_value < min_func2:
                best_func['2'] = [best_input, best_value]
                min_func2 = best_value
            elif func_num == 3 and best_value < min_func3:
                best_func['3'] = [best_input, best_value]
                min_func3 = best_value
            elif func_num == 4 and best_value < min_func4:
                best_func['4'] = [best_input, best_value]
                min_func4 = best_value
            
            #print('func'+str(func_num)+': ')
            #print('  best_input = {}, \n  best_value = {}\n'.format(best_input, best_value))

            # change the name of this file to your student_ID and it will output properlly
            with open("{}_function{}.txt".format('108065530'.split('_')[0], func_num), 'w+') as f:
                for i in range(op.dim):
                    f.write("{}\n".format(best_func[str(func_num)][0][i]))
                f.write("{}\n".format(best_func[str(func_num)][1]))
            func_num += 1
            
        try_times += 1
        #print('--------------------------------------------------------------------------------------------')
        
finish_time = datetime.datetime.now()
time_used = (finish_time - start_time).seconds / 60
print('time elapsed:', time_used, 'minutes')
