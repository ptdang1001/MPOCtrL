# -*- coding: utf8 -*

from collections import defaultdict

SEP_SIGN = "*" * 100

class MessagePassingOptimization: # message passing optimization
    def __init__(self, variables_dict, factors, variables, main_branch, args):
        self.factors = factors # factors contains the parent and child variables in dict format
        self.variables = variables # variables contains the parent and child factors in dict format
        self.main_branch = main_branch # main_branch contains the main branch of the graph
        self.args = args
        self.variables_dict = variables_dict
        self._msg_factors2variables = {}
        self._msg_variables2factors = {}
        self._variables_dict_new = {}

        self.visited_child_variables = defaultdict(dict)
        self.visited_parent_variables = defaultdict(dict)

        self._init_variable()
        self._init_msg_factors2variables()
        self._init_msg_variables2factors()

    def _init_variable(self):
        for factor_i, parent_child in self.factors.items():
            parent_variables = parent_child["parent_variables"]
            child_variables = parent_child["child_variables"]
            if len(parent_variables) == 1 and len(child_variables) == 1:
                continue

            if len(child_variables) == 1 and len(parent_variables) > 1:
                sum_parent_variables = sum(self.variables_dict[parent_variable_i] for parent_variable_i in parent_variables)
                self.variables_dict[child_variables[0]] = (sum_parent_variables + self.variables_dict[child_variables[0]]) / 2

            if len(parent_variables) == 1 and len(child_variables) > 1:
                sum_child_variables = sum(self.variables_dict[child_variable_i] for child_variable_i in child_variables)
                self.variables_dict[parent_variables[0]] = (sum_child_variables + self.variables_dict[parent_variables[0]]) / 2

    def get_variables_diff_factor2variable(self, cur_variable, parent_variables, child_variables):
        variables_sum_other_parent_variables = 0.0
        for parent_variable_i in parent_variables:
            if parent_variable_i == cur_variable: continue
            variables_sum_other_parent_variables += self.variables_dict[parent_variable_i]

        variables_sum_other_child_variables = 0.0
        for child_variable_i in child_variables:
            if child_variable_i == cur_variable: continue

            variables_sum_other_child_variables += self.variables_dict[child_variable_i]

        variables_difference = abs(
            variables_sum_other_parent_variables - variables_sum_other_child_variables
        )
        return variables_difference

    # @pysnooper.snoop()
    def _init_msg_factors2variables(self):
        for factor_i, parent_child in self.factors.items():
            parent_variables = parent_child["parent_variables"]
            child_variables = parent_child["child_variables"]

            for parent_variable_i in parent_variables:
                variables_difference = 0.0
                variables_difference = self.get_variables_diff_factor2variable(
                    parent_variable_i, parent_variables, child_variables
                )
                self._msg_factors2variables[(factor_i, parent_variable_i)] = variables_difference

            for child_variable_i in child_variables:
                variables_difference = 0.0
                variables_difference = self.get_variables_diff_factor2variable(
                    child_variable_i, parent_variables, child_variables
                )
                self._msg_factors2variables[(factor_i, child_variable_i)] = variables_difference

    # @pysnooper.snoop()
    def _init_msg_variables2factors(self):
        for variable_i, parent_child in self.variables.items():
            parent_factors = parent_child["parent_factors"]
            child_factors = parent_child["child_factors"]
            # all_neighbor_factors = parent_factors + child_factors

            for parent_factor_i in parent_factors:
                if (
                    variable_i,
                    parent_factor_i,
                ) in self._msg_variables2factors or parent_factor_i.startswith("dummy"):
                    continue

                self._msg_variables2factors[(variable_i, parent_factor_i)] = self.variables_dict[variable_i]

            for child_factor_i in child_factors:
                if (
                    variable_i,
                    child_factor_i,
                ) in self._msg_variables2factors or child_factor_i.startswith("dummy"):
                    continue

                # self.update_msg_variable2factor(variable_i, child_factor_i, all_neighbor_factors)
                self._msg_variables2factors[(variable_i, child_factor_i)] = self.variables_dict[variable_i]

    # @pysnooper.snoop()
    def get_factor_weight(self, variable, cur_factor, res_by="mean"):
        return 1.0

    # @pysnooper.snoop()
    def update_msg_factor2variable(self, factor, cur_variable, parent_variables, child_variables, flag):
        msg_from_parent_variables = 0.0  # exclude current variable
        msg_from_parent_variables = sum(
                [self._msg_variables2factors[(parent_variable_i, factor)]
                for parent_variable_i in parent_variables if parent_variable_i != cur_variable] + [0]
            )

        msg_from_child_variables = 0.0
        msg_from_child_variables = sum(
                [self._msg_variables2factors[(child_variable_i, factor)]
                for child_variable_i in child_variables if child_variable_i != cur_variable] + [0]
            )

        msg_factor2variable = msg_from_child_variables - msg_from_parent_variables
        negative_keep_ratio = 0.1

        if msg_from_parent_variables == 0:
            msg_factor2variable = msg_factor2variable
        elif msg_from_child_variables == 0:
            msg_factor2variable = -msg_factor2variable
        elif (
            msg_from_child_variables != 0
            and msg_from_parent_variables != 0
            and msg_factor2variable > 0
            and flag == "parent_variable"
        ):
            msg_factor2variable = msg_factor2variable
        elif (
            msg_from_child_variables != 0
            and msg_from_parent_variables != 0
            and msg_factor2variable > 0
            and flag == "child_variable"
        ):
            # msg_factor2variable is < 0
            # msg_factor2variable = -msg_factor2variable
            msg_factor2variable = (1.0 - negative_keep_ratio) * self._msg_variables2factors[
                (cur_variable, factor)
            ] + negative_keep_ratio * (-msg_factor2variable)
        elif (
            msg_from_child_variables != 0
            and msg_from_parent_variables != 0
            and msg_factor2variable < 0
            and flag == "parent_variable"
        ):
            # msg_factor2variable is < 0
            # msg_factor2variable = msg_factor2variable
            msg_factor2variable = (1.0 - negative_keep_ratio) * self._msg_variables2factors[
                (cur_variable, factor)
            ] + negative_keep_ratio * msg_factor2variable
        elif (
            msg_from_child_variables != 0
            and msg_from_parent_variables != 0
            and msg_factor2variable < 0
            and flag == "child_variable"
        ):
            msg_factor2variable = -msg_factor2variable

        self._msg_factors2variables[(factor, cur_variable)] = msg_factor2variable


    # @pysnooper.snoop()
    def update_msg_factor2variables(self, factor, parent_variables, child_variables):
        # update the msg(1 facor->other variables(the factor's parent and child))
        for parent_variable_i in parent_variables:
            self.update_msg_factor2variable(
                factor, parent_variable_i, parent_variables, child_variables, flag="parent_variable"
            )

        for child_variable_i in child_variables:
            self.update_msg_factor2variable(
                factor, child_variable_i, parent_variables, child_variables, flag="child_variable"
            )

    def get_weights_4_msg(self, factors):
        factors_weights = {}

        if len(self.main_branch) == 0:
            weight = 1.0 / len(factors)
            factors_weights = {factor_i: weight for factor_i in factors}
            return factors_weights

        n_factors_in_main_branch = 0
        n_factors_outside_main_branch = 0
        for factor_i in factors:
            if factor_i in self.main_branch:
                n_factors_in_main_branch += 1
            else:
                n_factors_outside_main_branch += 1

        if n_factors_in_main_branch == 0:
            factors_weights = {
                factor_i: 1.0 / n_factors_outside_main_branch for factor_i in factors
            }
        elif n_factors_outside_main_branch == 0:
            factors_weights = {
                factor_i: 1.0 / n_factors_in_main_branch for factor_i in factors
            }
        else:
            weight_4_factors_in_main_branch = self.args.mpo_beta_2
            weight_4_factors_outside_main_branch = 1.0 - weight_4_factors_in_main_branch
            for factor_i in factors:
                if factor_i in self.main_branch:
                    factors_weights[factor_i] = weight_4_factors_in_main_branch
                else:
                    factors_weights[factor_i] = weight_4_factors_outside_main_branch

        return factors_weights

    def update_msg_variable2factor(self, variable, cur_factor, all_neighbor_factors, flag):
        msg_variable2factor = 0.0
        factors_weights = self.get_weights_4_msg(all_neighbor_factors)

        msg_variable2factor = sum(
            [self.get_factor_weight(variable, factor_i)
            * factors_weights[factor_i]
            * self._msg_factors2variables[(factor_i, variable)]
            for factor_i in all_neighbor_factors] + [0]
        )

        learning_rate = self.args.mpo_learning_rate
        msg_variable2factor = learning_rate * msg_variable2factor + (
            1 - learning_rate
        ) * self.variables_dict[variable]

        self._msg_variables2factors[(variable, cur_factor)] = msg_variable2factor

        if flag == "parent_factor":
            self.visited_child_variables[variable] = {"parent_factor": cur_factor}
        elif flag == "child_factor":
            self.visited_parent_variables[variable] = {"child_factor": cur_factor}

    def update_msg_variable2factors(self, variable, parent_factors, child_factors):
        # update the msg(1 node->other factors(the ndoe's parent and child))
        for parent_factor_i in parent_factors:
            if variable in self.visited_child_variables:
                self._msg_variables2factors[(variable, parent_factor_i)] = (
                    self._msg_variables2factors[
                        (variable, self.visited_child_variables[variable]["parent_factor"])
                    ]
                )
                continue

            self.update_msg_variable2factor(
                variable,
                parent_factor_i,
                parent_factors + child_factors,
                flag="parent_factor",
            )

        for child_factor_i in child_factors:
            #'''
            if variable in self.visited_parent_variables:
                self._msg_variables2factors[(variable, child_factor_i)] = (
                    self._msg_variables2factors[
                        (variable, self.visited_parent_variables[variable]["child_factor"])
                    ]
                )
                continue
            #'''
            self.update_msg_variable2factor(
                variable,
                child_factor_i,
                parent_factors + child_factors,
                flag="child_factor",
            )

    # update_variables
    def update_variables(self):
        for variable_i in self.variables_dict.keys():
            parent_child_factors = (
                self.variables[variable_i]["parent_factors"]
                + self.variables[variable_i]["child_factors"]
            )
            if len(parent_child_factors) == 0:
                print("Wrong Variables!")
                print(f"variable_i:{variable_i}")
                print(f"parent_factors:{self.variables[variable_i]['parent_factors']}")
                print(f"child_factors:{self.variables[variable_i]['child_factors']}")
                print(f"self.variables:{self.variables}")
            factors_weights = self.get_weights_4_msg(parent_child_factors)
            variables_new = sum(
                [self.get_factor_weight(variable_i, factor_i)
                * factors_weights[factor_i]
                * self._msg_factors2variables[(factor_i, variable_i)]
                for factor_i in parent_child_factors] + [0]
            )
            self.variables_dict[variable_i] = variables_new


    # @pysnooper.snoop()
    def run(self):
        # update the msg_factors2variables and msg_variables2factors
        for i in range(1, (self.args.mpo_n_epoch + 1)):
            # these visited sets are for cycles and anchors
            self.visited_parent_variables = None
            self.visited_child_variables = None
            self.visited_parent_variables = defaultdict(dict)
            self.visited_child_variables = defaultdict(dict)

            # update the msg factors to variables
            for (
                factor_i,
                parent_child_variables,
            ) in self.factors.items():  # update msg_factors2variables
                self.update_msg_factor2variables(factor_i, parent_child_variables[
                    "parent_variables"
                ], parent_child_variables["child_variables"])

            # update the msg variables to factors
            for (
                variable_i,
                parent_child_factors,
            ) in self.variables.items():  # update msg_variables2factors
                self.update_msg_variable2factors(variable_i, parent_child_factors["parent_factors"], parent_child_factors["child_factors"])

            # update  variables' variables
            self.update_variables()

            self._variables_dict_new[f"epoch_{i}"]=self.variables_dict.copy()
