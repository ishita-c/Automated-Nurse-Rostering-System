import sys
import pandas as pd
import numpy as np
import json
import time

sys.setrecursionlimit(1500)


start = time.process_time()

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
        self.has_rest = [0 for i in range(self.N)]
        self.shift_counts = [0 for i in range(3)]
        self.cur_day = -1
        self.max_r = (7*(N-m-a-e) - N)//N + 3
        
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
        if "A" in domain:
            ordered_domain.append("A")
        if "M" in domain:
            ordered_domain.append("M")
        if "E" in domain:
            ordered_domain.append("E")
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
        updated_shift_counts = self.shift_counts[:]
        if value == "M":
            updated_shift_counts[0] += 1
        elif value == "A":
            updated_shift_counts[1] += 1
        elif value == "E":
            updated_shift_counts[2] += 1
        updated_has_rest = self.has_rest[:]
        if value == "R":
            updated_has_rest[n] += 1
        if updated_has_rest[n] > self.max_r:
            return False
        if (d + 1) % 7 == 0:
            if updated_has_rest[n] == 0:
                return False
        if (num_assigned + 1) % self.N == 0:
            if (updated_shift_counts[0] != self.m) or (updated_shift_counts[1] != self.a) or (updated_shift_counts[2] != self.e):
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
                for i in range(self.N):
                    var_name = "N" + str(i) + "_" + str(d)
                    if var_name in updated_domains:
                        if "A" in updated_domains[var_name]:
                            updated_domains[var_name].remove("A")
                            if len(updated_domains[var_name]) == 0:
                                return -1
        elif value == "E":
            if self.shift_counts[2] == self.e:
                # delete "E" from all domains of cur_day
                for i in range(self.N):
                    var_name = "N" + str(i) + "_" + str(d)
                    if var_name in updated_domains:
                        if "E" in updated_domains[var_name]:
                            updated_domains[var_name].remove("E")
                            if len(updated_domains[var_name]) == 0:
                                return -1
        # inferences for next day of this nurse
        if d < self.D-1:
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

        if len(self.assignment) % self.N == 0: # assignment for previous day completed, reset shift_counts
            self.cur_day += 1
            store_shift_counts = self.shift_counts[:]
            self.shift_counts = [0 for i in range(3)]
        
        if len(self.assignment) % (7 * self.N) == 0: # assignment for previous week completed, reset has_rest
            store_has_rest = self.has_rest[:]
            self.has_rest = [0 for i in range(self.N)]

        var = self.select_unassigned_variable()
        # print(f"Assigning variable {var}")
        n = int(var[1:var.find("_")])
        d = int(var[var.find("_")+1:])
        ordered_domain = self.order_domain_value(var)
        # print(f"Domain of variable {var}: {ordered_domain}")

        for value in ordered_domain:
            if self.check_consistency(var, value):
                # print(f"{var}: {value} consistent with assignment")
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
                    # print(f"Assigned {var} = {value}")
                    self.domains = inferences
                    result = self.backtracking_search()
                    if result != -1:
                        return result
            if var in self.assignment:
                # print(f"Undo {var} = {value}")
                del self.assignment[var]
                if value == "M":
                    self.shift_counts[0] -= 1
                elif value == "A":
                    self.shift_counts[1] -= 1
                elif value == "E":
                    self.shift_counts[2] -= 1
                if value == "R":
                    self.has_rest[n] -= 1
                self.domains = store_domains # undo inferences
                self.domains[var] = store_var_domain # add back domain of var since it gets unassigned
        
        if len(self.assignment) % self.N == 0: # in case of failure, undo cur_day updation and undo resetting of shift_counts
            self.cur_day -= 1
            self.shift_counts = store_shift_counts
        
        if len(self.assignment) % (7 * self.N) == 0: # in case of failure, undo resetting of has_rest
            self.has_rest = store_has_rest

        # print("Failure")

        return -1

class csp_pref:

    def __init__(self, N, D, m, a, e, S):
        self.N = N
        self.D = D
        self.m = m
        self.a = a
        self.e = e
        self.S = S
        self.domains = {}
        for n in range(N):
            for d in range(D):
                self.domains["N" + str(n) + "_" + str(d)] = {"M", "A", "E", "R"}
        self.assignment = {}
        self.has_rest = [0 for i in range(self.N)]
        self.shift_counts = [0 for i in range(3)]
        self.cur_day = -1
        self.max_r = (7*(N-m-a-e) - N)//N + 3
        
    def select_unassigned_variable(self):
        most_constrained = None
        least_domain_size = None
        day = self.cur_day
        for i in range(self.N):
            var_name = "N" + str(i) + "_" + str(day)
            if var_name in self.domains:
                cur_domain_size = len(self.domains[var_name])
                if i < self.S:
                    least_domain_size = cur_domain_size
                    most_constrained = var_name
                else:
                # choose in case of ties
                    if least_domain_size == None or cur_domain_size < least_domain_size:
                        least_domain_size = cur_domain_size
                        most_constrained = var_name
        return most_constrained

    def order_domain_value(self, var_name):
        n = int(var_name[1:var_name.find("_")])
        domain = self.domains[var_name]
        ordered_domain = []
        if n < self.S:
            if "M" in domain:
                ordered_domain.append("M")
            if "E" in domain:
                ordered_domain.append("E")
            if "A" in domain:
                ordered_domain.append("A")
            if "R" in domain:
                ordered_domain.append("R")
        else:
            if "A" in domain:
                ordered_domain.append("A")
            if "M" in domain:
                ordered_domain.append("M")
            if "E" in domain:
                ordered_domain.append("E")
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
        updated_shift_counts = self.shift_counts[:]
        if value == "M":
            updated_shift_counts[0] += 1
        elif value == "A":
            updated_shift_counts[1] += 1
        elif value == "E":
            updated_shift_counts[2] += 1
        updated_has_rest = self.has_rest[:]
        if value == "R":
            updated_has_rest[n] += 1
        if updated_has_rest[n] > self.max_r:
            return False
        if (d + 1) % 7 == 0:
            if updated_has_rest[n] == 0:
                return False
        if (num_assigned + 1) % self.N == 0:
            if (updated_shift_counts[0] != self.m) or (updated_shift_counts[1] != self.a) or (updated_shift_counts[2] != self.e):
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
                for i in range(self.N):
                    var_name = "N" + str(i) + "_" + str(d)
                    if var_name in updated_domains:
                        if "A" in updated_domains[var_name]:
                            updated_domains[var_name].remove("A")
                            if len(updated_domains[var_name]) == 0:
                                return -1
        elif value == "E":
            if self.shift_counts[2] == self.e:
                # delete "E" from all domains of cur_day
                for i in range(self.N):
                    var_name = "N" + str(i) + "_" + str(d)
                    if var_name in updated_domains:
                        if "E" in updated_domains[var_name]:
                            updated_domains[var_name].remove("E")
                            if len(updated_domains[var_name]) == 0:
                                return -1
        # inferences for next day of this nurse
        if d < self.D-1:
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

        if len(self.assignment) % self.N == 0: # assignment for previous day completed, reset shift_counts
            self.cur_day += 1
            store_shift_counts = self.shift_counts[:]
            self.shift_counts = [0 for i in range(3)]
        
        if len(self.assignment) % (7 * self.N) == 0: # assignment for previous week completed, reset has_rest
            store_has_rest = self.has_rest[:]
            self.has_rest = [0 for i in range(self.N)]

        var = self.select_unassigned_variable()
        # print(f"Assigning variable {var}")
        n = int(var[1:var.find("_")])
        d = int(var[var.find("_")+1:])
        ordered_domain = self.order_domain_value(var)
        # print(f"Domain of variable {var}: {ordered_domain}")

        for value in ordered_domain:
            if self.check_consistency(var, value):
                # print(f"{var}: {value} consistent with assignment")
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
                    # print(f"Assigned {var} = {value}")
                    self.domains = inferences
                    result = self.backtracking_search()
                    if result != -1:
                        return result
            if var in self.assignment:
                # print(f"Undo {var} = {value}")
                del self.assignment[var]
                if value == "M":
                    self.shift_counts[0] -= 1
                elif value == "A":
                    self.shift_counts[1] -= 1
                elif value == "E":
                    self.shift_counts[2] -= 1
                if value == "R":
                    self.has_rest[n] -= 1
                self.domains = store_domains # undo inferences
                self.domains[var] = store_var_domain # add back domain of var since it gets unassigned
        
        if len(self.assignment) % self.N == 0: # in case of failure, undo cur_day updation and undo resetting of shift_counts
            self.cur_day -= 1
            self.shift_counts = store_shift_counts
        
        if len(self.assignment) % (7 * self.N) == 0: # in case of failure, undo resetting of has_rest
            self.has_rest = store_has_rest

        # print("Failure")

        return -1

if values.size == 5:
    csp_solver = csp(values[0, 0], values[0, 1], values[0, 2], values[0, 3], values[0, 4])
    if (values[0, 2] + values[0, 3] + values[0, 4] > values[0, 0]) or ((values[0, 2] + values[0, 3] + values[0, 4] == values[0, 0]) and values[0,1] >= 7):
        print("NO-SOLUTION")
        assignment = {}
    else:
        assignment = csp_solver.backtracking_search()
        if assignment == -1:
            print("NO-SOLUTION")
            assignment = {}
        else:
            # print(assignment)
            print("SOLUTION")
            for i in range(csp_solver.N):
                print(f"N-{i}", end=" ")
                for j in range(csp_solver.D):
                    print(assignment["N"+str(i)+"_"+str(j)], end=" ")
                print("")
    soln_list = [assignment]
    with open("solution.json", 'w') as file:
        for d in soln_list:
            json.dump(d, file)
            file.write("\n")

elif values.size == 7:
    csp_solver = csp_pref(values[0, 0], values[0, 1], values[0, 2], values[0, 3], values[0, 4], values[0,5])
    if (values[0, 2] + values[0, 3] + values[0, 4] > values[0, 0]) or ((values[0, 2] + values[0, 3] + values[0, 4] == values[0, 0]) and values[0,1] >= 7):
        print("NO-SOLUTION")
        assignment = {}
    else:
        assignment = csp_solver.backtracking_search()
        if assignment == -1:
            print("NO-SOLUTION")
            assignment = {}
        else:
            # print(assignment)
            print("SOLUTION")
            for i in range(csp_solver.N):
                print(f"N-{i}", end=" ")
                for j in range(csp_solver.D):
                    print(assignment["N"+str(i)+"_"+str(j)], end=" ")
                print("")
    soln_list = [assignment]
    with open("solution.json", 'w') as file:
        for d in soln_list:
            json.dump(d, file)
            file.write("\n")

else:
    raise Exception("Improper number of inputs in file.")

time_taken = time.process_time() - start
print("Time Taken:", time_taken)
