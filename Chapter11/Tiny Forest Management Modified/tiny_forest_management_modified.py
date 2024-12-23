import mdptoolbox.example


class ForestMDP:
    def __init__(self, num_states, num_actions, num_trees, prob_success, gamma=0.9):
        self.P, self.R = mdptoolbox.example.forest(num_states, num_actions, num_trees, prob_success)
        self.gamma = gamma
        self.policy_iteration_model = None

    def display_transition_probabilities(self):
        print("Transition Probability Matrix for Action 0:")
        print(self.P[0])
        print("Transition Probability Matrix for Action 1:")
        print(self.P[1])

    def display_reward_matrix(self):
        print("Reward Matrix for Action 0:")
        print(self.R[:, 0])
        print("Reward Matrix for Action 1:")
        print(self.R[:, 1])

    def run_policy_iteration(self):
        self.policy_iteration_model = mdptoolbox.mdp.PolicyIteration(self.P, self.R, self.gamma)
        self.policy_iteration_model.run()

    def display_results(self):
        if self.policy_iteration_model is not None:
            print("Value Function:")
            print(self.policy_iteration_model.V)
            print("Optimal Policy:")
            print(self.policy_iteration_model.policy)
            print("Number of Iterations:")
            print(self.policy_iteration_model.iter)
            print("Time taken (seconds):")
            print(self.policy_iteration_model.time)
        else:
            print("Policy Iteration has not been run yet.")


if __name__ == "__main__":
    forest_mdp = ForestMDP(num_states=3, num_actions=4, num_trees=2, prob_success=0.8)
    forest_mdp.display_transition_probabilities()
    forest_mdp.display_reward_matrix()
    forest_mdp.run_policy_iteration()
    forest_mdp.display_results()
