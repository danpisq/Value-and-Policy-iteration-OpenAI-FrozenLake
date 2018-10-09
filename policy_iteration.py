"""Brute force"""

import numpy 
import time 
import gym

"""
    Args:
    poicy [S,A] shaped matrix representing policy.
    env. OpenAi gym env.v.
      env.P represents the transition propablities of the env
      env.P[s][a] is a list of transition tuples 
      env.nS = is a number of states
      env.nA is a number of actions
    gamma: discount factor
    render: boolean to turn rendering on/off 
"""

def execute(env, policy, gamma=1.0, render=False):
  start = env.reset()
  totalReward = 0
  stepIndex = 0
  while True:
    if render:
      env.render()
    start, reward, done,_ = env.step(int(policy[start]))
    totalReward += (gamma ** stepIndex * reward)
    stepIndex += 1
    if done:
      break
  return totalReward
    
#Evaluation
def evaluatePolicy(env, policy, gamma=1.0, n=100):
  scores = [execute(env, policy, gamma, False) for _ in range(n)]
  return numpy.mean(scores)
  
#Extract the policy given a value-function
def extractPolicy(v, gamma=1.0):
  policy = numpy.zeros(env.env.nS)
  for s in range(env.env.nS):
    q_sa = numpy.zeros(env.env.nA)
    for a in range(env.env.nA):
      q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])
      policy[s] = numpy.argmax(q_sa)
  return policy
      
#Iteratively calculates value-function under policy   
def CalcPolicyValue(env, policy, gamma=1.0):
  value = numpy.zeros(env.env.nS)
  eps = 0.1
  while True:
    previousValue = numpy.copy(value)
    for states in range(env.env.nS):
      policy_a = policy[states]
      value[states] = sum([p * (r + gamma * previousValue[s_]) for p,s_, r, _ in env.env.P[states][policy_a]])
    if (numpy.sum((numpy.fabs(previousValue - value))) <= eps):
      break
      print( "breaked")
  return value
  
  
#Policy Iteration algorithm
def policyIteration(env, gamma=1.0):
  policy = numpy.random.choice(env.env.nA, size=(env.env.nS))
  maxIterations = 1000
  gamma = 1.0
  for i in range(maxIterations):
    oldPolicyValue = CalcPolicyValue(env, policy, gamma)
    newPolicy = extractPolicy(oldPolicyValue, gamma)
    if (numpy.all(policy == newPolicy)):
      print('Policy Iteration converged at %d' %(i+1))
      break
    policy = newPolicy
  return policy


if __name__ == '__main__':
  env = gym.make('FrozenLake-v0')
  ## Policy search
  startTime = time.time()
  optimalPolicy = policyIteration(env, gamma = 1.0)
  scores = evaluatePolicy(env, optimalPolicy, gamma=1.0)
  endTime = time.time()
  print("Best score = %0.2f. Time taken = %4.4f seconds" %(numpy.max(scores), endTime-startTime))
  print(scores)
