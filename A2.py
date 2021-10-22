import sys
import pandas as pd
import numpy as np


if len(sys.argv) != 2:
    raise Exception("Please give filename as argument.")

filename = sys.argv[1]

dataframe = pd.read_csv(filename)
values = dataframe.to_numpy(dtype=np.int64)

class csp:

    def __init__(self, N, D, m, a, e):
        self.N = N
        self.D = D
        self.m = m
        self.a = a
        self.e = e
        self.domains = {}
        for n in range(N):
            for d in range(D):
                self.domains["N" + str(n) + "_" + str(d)] = {"M", "A", "E", "R"}
        self.assignment = {}
        self.has_rest = [0 in range(N)]
        self.shift_counts = [0 in range(3)]
        self.cur_day = -1   
        
    def select_unassigned_variable(self):
        most_constrained = None
        least_domain_size = None
        day = self.cur_day
        for i in range(self.N):
            var_name = "N" + str(i) + "_" + str(day)
            if var_name in self.domains:
                cur_domain_size = len(self.domains[var_name])
                # choose in case of ties
                if least_domain_size == None or cur_domain_size < least_domain_size:
                    least_domain_size = cur_domain_size
                    most_constrained = var_name
        return most_constrained

    def order_domain_value(self, var_name):
        domain = self.domains[var_name]
        ordered_domain = []
        if "M" in domain:
            ordered_domain.append("M")
        if "E" in domain:
            ordered_domain.append("E")
        if "A" in domain:
            ordered_domain.append("A")
        if "R" in domain:
            ordered_domain.append("R")
        return ordered_domain

    def check_consistency(self, var, value):
        '''
            checks consistency of var = value with current partial assignment
        '''
        n = int(var[1:var.find("_")])
        d = int(var[var.find("_")+1:])
        if d > 0 and value == "M":
            prev_shift = self.assignment["N" + str(n) + "_" + str(d-1)]
            if prev_shift == "M" or prev_shift == "E":
                return False
        num_assigned = len(self.assignment)
        print(self.shift_counts)
        updated_shift_counts = self.shift_counts[:]
        print(updated_shift_counts)
        if value == "M":
            updated_shift_counts[0] += 1
        elif value == "A":
            updated_shift_counts[1] += 1
        elif value == "E":
            updated_shift_counts[2] += 1
        updated_has_rest = self.has_rest[:]
        if value == "R":
            updated_has_rest[n] += 1
        if (num_assigned + 1) % self.D == 0:
            if (updated_shift_counts[0] != self.m) or (updated_shift_counts[1] != self.a) or (updated_shift_counts[2] != self.e):
                return False            
        if (num_assigned + 1) % (7 * self.D * self.N) == 0:
            for nurse_rest in updated_has_rest:
                if nurse_rest == 0:
                    return False
        return True

    def get_inferences(self, var):
        '''
            returns updated 
        '''
        updated_domains = {key: value.copy() for key, value in self.domains.items()}
        n = int(var[1:var.find("_")])
        d = int(var[var.find("_")+1:])
        value = self.assignment[var]
        # inferences from m, a, e constraint
        if value == "M":
            if self.shift_counts[0] == self.m:
                # delete "M" from all domains of cur_day
                for i in range(self.N):
                    var_name = "N" + str(i) + "_" + str(d)
                    if var_name in updated_domains:
                        if "M" in updated_domains[var_name]:
                            updated_domains[var_name].remove("M")
                            if len(updated_domains[var_name]) == 0:
                                return -1
        elif value == "A":
            if self.shift_counts[1] == self.a:
                # delete "A" from all domains of cur_day
                for i in range(N):
                    var_name = "N" + str(i) + "_" + str(d)
                    if var_name in updated_domains:
                        if "A" in updated_domains[var_name]:
                            updated_domains[var_name].remove("A")
                            if len(updated_domains[var_name]) == 0:
                                return -1
        elif value == "E":
            if self.shift_counts[2] == self.e:
                # delete "E" from all domains of cur_day
                for i in range(N):
                    var_name = "N" + str(i) + "_" + str(d)
                    if var_name in updated_domains:
                        if "E" in updated_domains[var_name]:
                            updated_domains[var_name].remove("E")
                            if len(updated_domains[var_name]) == 0:
                                return -1
        # inferences for next day of this nurse
        if d < self.D:
            var_name = "N" + str(n) + "_" + str(d+1)
            if value == "M" or value == "E":
                updated_domains[var_name].remove("M")
                if len(updated_domains[var_name]) == 0:
                                return -1
        return updated_domains

    def backtracking_search(self):
        '''
            returns a valid assignment, or -1 in case of no solution
        '''
        if len(self.assignment) == self.N * self.D:
            return self.assignment

        if len(self.assignment) % self.N == 0: # assignment for previous day completed
            self.cur_day += 1

        var = self.select_unassigned_variable()
        n = int(var[1:var.find("_")])
        d = int(var[var.find("_")+1:])
        ordered_domain = self.order_domain_value(var)

        for value in ordered_domain:
            if self.check_consistency(var, value):
                self.assignment[var] = value
                if value == "M":
                    self.shift_counts[0] += 1
                elif value == "A":
                    self.shift_counts[1] += 1
                elif value == "E":
                    self.shift_counts[2] += 1
                if value == "R":
                    self.has_rest[n] += 1
                store_var_domain = self.domains[var].copy()
                del self.domains[var]
                store_domains = {key: value.copy() for key, value in self.domains.items()}
                inferences = self.get_inferences(var)
                if inferences != -1:
                    self.domains = inferences
                    result = self.backtracking_search()
                    if result != -1:
                        return result
            del self.assignment[var]
            if value == "M":
                self.shift_counts[0] -= 1
            elif value == "A":
                self.shift_counts[1] -= 1
            elif value == "E":
                self.shift_counts[2] -= 1
            if value == "R":
                self.has_rest[n] -= 1
            self.domains[var] = store_var_domain
            self.domains = store_domains
        
        if len(self.assignment) % self.N == 0: # undo cur_day updation in case of failure
            self.cur_day -= 1

        return -1       

if values.size == 5:
    csp_solver = csp(values[0, 0], values[0, 1], values[0, 2], values[0, 3], values[0, 4])
    assignment = csp_solver.backtracking_search()
    if assignment == -1:
        print("NO-SOLUTION")
        assignment = {}
    else:
        print(assignment)
    with open("solution.json", 'w') as file:
        for d in assignment:
            json.dump(d, file)
            file.write("\n")


# elif values.size == 7:

else:
    raise Exception("Improper number of inputs in file.")