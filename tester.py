import numpy as np
import torch
import gym
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys


import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
#from policy_model import Policy


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
    
    

class Tester(object):

    def __init__(self):
        """
        Initialize the Tester object by loading your model.
        """
        # TODO: Load your pyTorch model for Policy Gradient here.
        
        self.model = Policy()
        self.model.load_state_dict(torch.load("./policy_mod.pt"))
        self.model.eval()
        
        pass


    def evaluate_policy(self, env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
        """Evaluate the value of a policy.

        See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        policy: np.array
          The policy to evaluate. Maps states to actions.
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.

        Returns
        -------
        np.ndarray
          The value for the given policy
        """
        # TODO: Your Code Goes Here
        value = np.zeros([16,1])
        
        R = np.zeros([16,1])
        P = np.zeros([16,16])
        
        for j in range(16):
            reward_mat = np.array(env.P[j][policy[j]])
            R[j,0] = np.sum(reward_mat[:,0]*reward_mat[:,2])
            for m in range(reward_mat.shape[0]):
                P[j,reward_mat[m,1].astype(int)] += reward_mat[m,0] 
        for i in range(max_iterations):
            prev_value = np.copy(value)
            value = R + gamma*np.matmul(P,value)
            #if (np.all(np.abs(prev_value-value) < tol)):
            if (np.linalg.norm(prev_value-value)<tol):
                break
        return value.reshape((16,)),i

    def policy_iteration(self, env, gamma, max_iterations=int(1e3), tol=1e-3):
        """Runs policy iteration.

        See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        You should use the improve_policy and evaluate_policy methods to
        implement this method.

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.

        Returns
        -------
        (np.ndarray, np.ndarray, int, int)
           Returns optimal policy, value function, number of policy
           improvement iterations, and number of value iterations.
        """
        # TODO:  Your code goes here.
        
        
        
        R_a = np.zeros([16,4])
        P_0 = np.zeros([16,16])
        P_1 = np.zeros([16,16])
        P_2 = np.zeros([16,16])
        P_3 = np.zeros([16,16])
        opt_policy = np.random.randint(4, size=16)
        #policy = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]).T
        #opt_policy = np.copy(policy)
        for j in range(16):
            for k in range(4):
                reward_mat = np.array(env.P[j][k])
                R_a[j,k] = np.sum(reward_mat[:,0]*reward_mat[:,2])
                if k==0:
                    for m in range(reward_mat.shape[0]):
                        P_0[j,reward_mat[m,1].astype(int)] += reward_mat[m,0]
                if k==1:
                    for m in range(reward_mat.shape[0]):
                        P_1[j,reward_mat[m,1].astype(int)] += reward_mat[m,0]
                if k==2:
                    for m in range(reward_mat.shape[0]):
                        P_2[j,reward_mat[m,1].astype(int)] += reward_mat[m,0]
                if k==3:
                    for m in range(reward_mat.shape[0]):
                        P_3[j,reward_mat[m,1].astype(int)] += reward_mat[m,0]
                    
        for i in range(max_iterations):
            previous_policy = np.copy(opt_policy)
            val,it = self.evaluate_policy(env, gamma, opt_policy, max_iterations, tol)
            value = val.reshape((16,1))
            P_all = np.append(np.matmul(P_0,value),np.matmul(P_1,value),axis=1)
            P_all = np.append(P_all,np.matmul(P_2,value),axis=1)
            P_all = np.append(P_all,np.matmul(P_3,value),axis=1)
            q_sa = R_a + gamma*(P_all)
            opt_policy = np.argmax(q_sa,axis=1)
            print(opt_policy)
            if(np.all(previous_policy==opt_policy)):
                break
            
        return opt_policy.reshape((16,)), value.reshape((16,)), i, it

    def value_iteration(self, env, gamma, max_iterations=int(1e3), tol=1e-3):
        """Runs value iteration for a given gamma and environment.

        See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.

        Returns
        -------
        np.ndarray, iteration
          The value function and the number of iterations it took to converge.
        """
        # TODO: Your Code goes here.
        R_a = np.zeros([16,4])
        P_0 = np.zeros([16,16])
        P_1 = np.zeros([16,16])
        P_2 = np.zeros([16,16])
        P_3 = np.zeros([16,16])
        #opt_policy = np.random.randint(4, size=16)
        #policy = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]).T
        #opt_policy = np.copy(policy)
        for j in range(16):
            for k in range(4):
                reward_mat = np.array(env.P[j][k])
                R_a[j,k] = np.sum(reward_mat[:,0]*reward_mat[:,2])
                if k==0:
                    for m in range(reward_mat.shape[0]):
                        P_0[j,reward_mat[m,1].astype(int)] += reward_mat[m,0]
                if k==1:
                    for m in range(reward_mat.shape[0]):
                        P_1[j,reward_mat[m,1].astype(int)] += reward_mat[m,0]
                if k==2:
                    for m in range(reward_mat.shape[0]):
                        P_2[j,reward_mat[m,1].astype(int)] += reward_mat[m,0]
                if k==3:
                    for m in range(reward_mat.shape[0]):
                        P_3[j,reward_mat[m,1].astype(int)] += reward_mat[m,0]
        
        opt_value = np.zeros([16,1])            
        for i in range(max_iterations):
            previous_value = np.copy(opt_value.reshape((16,1)))
            #value = self.evaluate_policy(env, self.gamma, self.policy, max_iterations=int(1e3), tol=1e-3)
            P_all = np.append(np.matmul(P_0,opt_value.reshape((16,1))),np.matmul(P_1,opt_value.reshape((16,1))),axis=1)
            P_all = np.append(P_all,np.matmul(P_2,opt_value.reshape((16,1))),axis=1)
            P_all = np.append(P_all,np.matmul(P_3,opt_value.reshape((16,1))),axis=1)
            q_sa = R_a + gamma*(P_all)
            opt_value = np.max(q_sa,axis=1)
            if(np.linalg.norm(previous_value-opt_value)<=tol):
                break
            
            
        return opt_value.reshape((16,)), i

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.model.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def policy_gradient_test(self, state):
        """
        Parameters
        ----------
        state: np.ndarray
            The state from the CartPole gym environment.
        Returns
        ------
        np.ndarray
            The action in this state according to the trained policy.
        """
        # TODO. Your Code goes here.
        #env = gym.make('CartPole-v0')
        action = self.select_action(state)
        #state, reward, done, _ = env.step(action)
        
        return action