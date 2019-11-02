import numpy as np
from anytree import Node
import pandas as pd

if __name__!="__main__":
    class MDP:
        def __init__(self):
            self.end = Node("11am",prob={})
            self.TU10a = Node("TU10a",prob={'P':((1,self.end,-1),),'R':((1,self.end,-1),),'S':((1,self.end,-1),)})
            self.RU10a = Node("RU10a",prob={'P':((1,self.end,0),),'R':((1,self.end,0),),'S':((1,self.end,0),)})
            self.RD10a = Node("RD10a",prob={'P':((1,self.end,4),),'R':((1,self.end,4),),'S':((1,self.end,4),)})
            self.TD10a = Node("TD10a",prob={'P':((1, self.end,3),),'R':((1,self.end,3),),'S':((1,self.end,3),)})
            self.RU8a = Node("RU8a",prob={'P':((1,self.TU10a,2),),'R':((1,self.RU10a,0),),'S':((1,self.RD10a,-1),)})
            self.RD8a = Node("RD8a",prob={'R':((1,self.RD10a,0),),'P':((1,self.TD10a,2),)})
            self.TU10p = Node("TU10p",prob={'P':((1,self.RU10a,2),),'R':((1,self.RU8a,0),)})
            self.RU10p = Node("RU10p",prob={'P':((0.5,self.RU8a,2),(0.5,self.RU10a,2)),'R':((1,self.RU8a,0),),'S':((1,self.RD8a,-1),)})
            self.RD10p = Node("RD10p",prob={'P':((0.5,self.RD8a,2),(0.5,self.RD10a,2)),'R':((1,self.RD8a,0),)})
            self.RU8p = Node("RU8p",prob={'P':((1,self.TU10p,2),),'R':((1,self.RU10p,0),),'S':((1,self.RD10p,-1),)})

        def actions(self): #returns possible actions for a given state
            return self.prob.keys()

        # def rewards(self,state,action):


    MarkovDP = MDP()
    # print(MarkovDP.actions(MarkovDP.RD10p)) #dict_keys(['P', 'R'])

    episodes = 50

    init_state = MarkovDP.RU8p

    print(init_state.prob['S'][0][0]) #prints the probability from init to next given action 'S'

if __name__=="__main__":
    class State():
        def __init__(self, name, actions, next_states, probs, rewards):
            self.name = name
            self.actions = actions
            self.next_states = next_states
            self.probs = probs
            self.rewards = rewards
            if len(self.actions) == 0:
                self.isendstate = True
            else:
                self.isendstate = False

        # def get_state_value(self): #float
        #     gamma = 1.0
        #     # print(gamma)
        #     print(self.actions)
        #     print(len(self.actions))
        #     # self.state = state
        #     if len(self.actions) == 0:
        #         self.tval = 0 #temporary value
        #     #     # break
        #     else:
        #         # print("Here")
        #         uAct = len(set(self.actions)) #unique actions
        #         # print("#Uniques:",uAct)
        #         eq_prob = 1.0/uAct
        #     #     self.tval = np.zeros(len(self.actions))
        #         for i in range(len(self.actions)):
        #             nextVs = self.next_states                    #nextVs is a list object
        #             # nextVs = State.get_state_value(self.next_states)
        #             print(nextVs)
        #     #         self.tval[i] = eq_prob * self.probs[i] * (self.rewards[i]*nextVs)
        #     # self.val = np.sum(self.tval)

if __name__=="__main__":
    ending = State('11a', [], [], [], [])
    TU10a = State('TU10a', ['P', 'R', 'S'], [ending, ending, ending], [1., 1., 1.], [-1., -1., -1.])
    RU10a = State('RU10a', ['P', 'R', 'S'], [ending, ending, ending], [1., 1., 1.], [0., 0., 0.])
    RD10a = State('RD10a', ['P', 'R', 'S'], [ending, ending, ending], [1., 1., 1.], [4., 4., 4.])
    TD10a = State('TD10a', ['P', 'R', 'S'], [ending, ending, ending], [1., 1., 1.], [3., 3., 3.])

    RU8a = State('RU8a', ['P', 'R', 'S'], [TU10a, RU10a, RD10a], [1., 1., 1.], [2., 0., -1.])
    RD8a = State('RD8a', ['P', 'R'], [TD10a, RD10a], [1., 1.], [2., 0.])

    TU10p = State('TU10p', ['P', 'R'], [RU10a, RU8a], [1., 1.], [2., 0.])
    # RU10p = State('RU10p',['R','P','S'],[RU8a,[RU8a,RU10a],RD8a],[1,[0.5,0.5],1],[0,[2,2],-1])
    RU10p = State('RU10p', ['P', 'P', 'R', 'S'], [RU8a, RU10a, RU8a, RD8a], [0.5, 0.5, 1., 1.], [2., 2., 0., -1.])
    # RD10p = State('RD10p',['R','P'],[RD8a,[RD8a,RD10a]],[1,[0.5,0.5]],[0,[2,2]])
    RD10p = State('RD10p', ['P', 'P', 'R'], [RD8a, RD10a, RD8a], [0.5, 0.5, 1.], [2., 2., 0.])
    RU8p = State('RU8p', ['P', 'R', 'S'], [TU10p, RU10p, RD10p], [1., 1., 1.], [2., 0., -1.])

states = [RU8p, TU10p, RU10p, RD10p, RU8a, RD8a, TU10a, RU10a, RD10a, TD10a, ending]

if __name__!="__main__":
    episodes = 50
    totsum = 0
    table = [['#Episode', 'States', 'Actions', 'Rewards', 'Return']]
    for i in range(episodes):
        run = True
        line = []
        state_list = []
        act_list = []
        rew_list = []
        state = RU8p
        while (run):
            # nAct = len(state.actions) #len(ending.actions) = 0
            if state.isendnode == False:
                nAct = len(state.actions)  # len(ending.actions) = 0
                state_list.append(state.name)
                choose = np.int(np.random.randint(0, nAct, 1))
                act_list.append(state.actions[choose])
                rew_list.append(state.rewards[choose])
                # print(state.name)
                state = state.next_states[choose]
                # print(state.name)
            else:
                run = False
        line.append([i+1,state_list,act_list,rew_list,np.sum(rew_list)])
        table = np.vstack((table,line))

        # exp_list.append()
        totsum += np.sum(rew_list)
        # print("Episode: ", i + 1)
        # print("States: ", state_list)
        # print("Actions: ", act_list)
        # print("Rewards: ", rew_list)
        # print("Return in episode: ", np.sum(rew_list))
        # print("\n")
    # print(pd.DataFrame(table))
    df = pd.DataFrame(table)
    export_CSV =df.to_csv(r'C:\Users\ranab\OneDrive\PycharmProjects\RL_assignment\Episode_table.csv')
    print("Average return for the {} episodes is: {}".format(episodes, totsum / 50))


#RU8p value 3.51

if __name__!="__main__":

    def get_state_value(gamma,state):
        nAct = len(state.actions)
        state.val = np.zeros(nAct)
        uAct = len(set(state.actions)) #will remove duplicate values, just uniques
        if nAct == 0:
            state.val = 0.
        else:
            eq_prob = 1.0 / uAct

            for i in range(nAct):
                if state.name == '11a':
                    nextVs = 0
                else:
                    nextVs = get_state_value(gamma,state.next_states[i])
                state.val[i] = eq_prob * state.probs[i]*(state.rewards[i]+(gamma * nextVs))
        return np.sum(state.val)
    i=0
    row = [['No.','States','Values']]
    for s in states:
        i+=1
        line = []
        value = get_state_value(1.0,s)
        line.append([i,s.name,value])

        row = np.vstack((row,line))
    print(pd.DataFrame(row))
    df = pd.DataFrame(row)
    export_CSV=df.to_csv(r'C:\Users\ranab\OneDrive\PycharmProjects\RL_assignment\state_values.csv')
    # value = get_state_value(1.0,RU8p)
    # print(value)

"""%%%% 3B %%%%"""
# states = [RU8p,TU10p,RU10p,RD10p,RU8a,RD8a,TU10a,RU10a,RD10a,TD10a,ending]
Vinit = {RU8p:0, TU10p:0,RU10p:0,RD10p:0,RU8a:0,RD8a:0,TU10a:0,RU10a:0,RD10a:0,TD10a:0,ending:0}
PartyPolicy = {RU8p:'P', TU10p:'P',RU10p:'P',RD10p:'P',RU8a:'P',RD8a:'P',TU10a:'P',RU10a:'P',RD10a:'P',TD10a:'P',ending:'P'}
# print(Vinit[RU8p])

# cnd = True
# gamma = 1.0
# iter = 0

# row = []
# for s in states:
#     row.append([s.name])
# # row = [row]
# print(row,len(row)) #[[#10]],1

# if __name__=="__main__":

def Evaluation(Policy,Vinit,states):
    theta = 0.001
    gamma = 1.0
    cnd = True
    iter = 0
    row = []
    # lst = []
    # row = [[]]
    # for s in states:
    #     row.append(s.name)
    while(cnd):
        iter += 1
        lstAc=[]
        lstVal=[]
        # print('Iteration:', iter)
        # col =
        delta = 0.0 #initially change in state value = 0
        for s in states:
            # iter += 1
            # print(s.name)
            v = Vinit[s] #initially 0
            # print(v)
            a = Policy[s] #P type:str
            # print(a,type(a))
            sum = []
            for j in range(len(s.actions)):
                # print(s.name)
                # print(":",j) #int
                if s.actions[j] == a: #1p 2p
                    sum.append(s.probs[j]*(s.rewards[j]+gamma*Vinit[s.next_states[j]]))
            Vinit[s] = np.sum(sum)
            # print("{}".format(s.name), Vinit[s])
            delta = max(delta,abs(v-Vinit[s]))
            # print('Delta:',delta)
            lstAc.append([a])
            lstVal.append([Vinit[s]])
        # lst =[lst]
            # rows = np.hstack((lst))
        # print('t',lst)
        # print('len(lst):',len(lst),len(row))
        # row.append(((lst)),axis=0)#vstack((row,lst))
        # row = np.hstack((row,lstAc,lstVal))
        cnd = (delta >= theta)
        # if delta < 0.001:
        #     cnd = False

    # print(iter)
    # print(Vinit)
    # print('delta:',delta)
    # s = 0
    # print(states[0])
    # s = states[0]
    # print(s)
    # a = PartyPolicy[s]
    # print(len(s.actions))
    # print(s.actions[0]==a)
    # print(pd.DataFrame(row))
    # print(Vinit)

    return Vinit
    # df = pd.DataFrame(row)
    # save_csv =df.to_csv(r'C:\Users\ranab\OneDrive\PycharmProjects\RL_assignment\pol_eval.csv')


def get_policy_actions_or_values(Policy): #or Vinit
    Action_list = []
    for i in Policy:
        Action_list.append(Policy[i])
    return Action_list


Policy = {RU8p:'P',TU10p:'P',RU10p:'P',RD10p:'P',RU8a:'P',RD8a:'P',TU10a:'P',RU10a:'P',RD10a:'P',TD10a:'P',ending:'P'}

Vinit = Evaluation(PartyPolicy,Vinit,states)

iterations = 0

if __name__ == "__main__":
    policy_stable = True
    gamma = 1.0
    proof = 0
    while(policy_stable): #policy is stable
        iterations += 1
        print('Iterations:', iterations)
        check_action = 0
        print(get_policy_actions_or_values(Policy))
        print(get_policy_actions_or_values(Vinit))
        for s in states:
            if s.isendnode == False: #not end state
                old_action = Policy[s] #this is the starting policy
                # sum = []
                uAct = len(set(s.actions)) #3
                # uActList = sorted(list(set(s.actions))) #['P', 'R', 'S']
                # a = 'P'
                # tsum = np.zeros(len(s.actions))
                # tsum=[]
                # for j in len(s.actions): #3
                #     if uAct == len(s.actions): #no dual actions
                #         tsum[j] = s.probs[j]*(s.rewards[j]+gamma*Vinit[s.next_states[j]])
                #     else:
                #         for k in range(2):
                if uAct == len(s.actions):  # no dual actions
                    tsum = []
                    for j in range(len(s.actions)):
                        # print("Singles",j)
                        tsum.append(s.probs[j]*(s.rewards[j]+gamma*Vinit[s.next_states[j]]))
                    u = -1 #just to find max
                    maximum = max(tsum)
                    for ii in tsum:
                        u += 1
                        if ii == maximum:
                            break

                    new_action = s.actions[u]
                    if old_action == new_action:
                        check_action+=1
                    else:
                        Policy[s] = new_action
                        Vinit = Evaluation(Policy, Vinit, states)
                    # Policy[s] = s.actions[u]
                    # Vinit[s] = maximum #shall we take the maximum value?

                else: #dual actions
                    ttsum = np.zeros(uAct)
                    dual_sum = []
                    for j in range(len(s.actions)):
                        # print("Doubles:",j)
                        if j<2:
                            dual_sum.append(s.probs[j]*(s.rewards[j]+gamma*Vinit[s.next_states[j]]))
                            d = 0
                            ttsum[d] = np.sum(dual_sum)
                        else:
                            d+=1
                            ttsum[d] = s.probs[j]*(s.rewards[j]+gamma*Vinit[s.next_states[j]])
                    u = -1  # just to find max
                    maximum = max(ttsum)
                    for ii in ttsum:
                        u += 1
                        if ii == maximum:
                            break
                    # Policy[s] = s.actions[u] #updating policy
                    new_action = s.actions[u]
                    if old_action==new_action:
                        # policy_stable = True
                        check_action += 1
                    else:
                        Policy[s] = new_action
                        Vinit = Evaluation(Policy,Vinit,states)

                # print(Policy[s])
                # print(Vinit[s])
        # print("#States with same new action:",check_action)
                # Vinit[s] = maximum #updating value
        # for p in Policy:
        if check_action == 10:
            proof +=1
        if proof == 2:
            policy_stable = False




    # print(Policy)

            # if s.actions[j] == 'P':
            #     sum.append(s.probs[j] * (s.rewards[j] + gamma * Vinit[s.next_states[j]])) #append the P actions
            #     d = 1
            # else:
            #     d+=1
            #     sum = s.probs[j] * (s.rewards[j] + gamma * Vinit[s.next_states[j]])
            #
            #
            #
            # a = s.actions[j]
            #
            #
            #
            #
            #
            # sum = np.sum(sum)





