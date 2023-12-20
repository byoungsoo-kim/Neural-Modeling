import numpy as np
from scipy import stats # for gaussian noise
from environment import Environment
from environment import Environment_TwoStepAgent

class DynaAgent(Environment):

    def __init__(self, alpha, gamma, epsilon):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha   -- learning rate \in (0, 1]
            gamma   -- discount factor \in (0, 1)
            epsilon -- controls the influence of the exploration bonus
        '''

        self.alpha   = alpha
        self.gamma   = gamma 
        self.epsilon = epsilon

        return None

    def init_env(self, **env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''

        Environment.__init__(self, **env_config)

        return None

    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''

        self.Q = np.zeros((self.num_states, self.num_actions))

        return None

    def _init_experience_buffer(self):

        '''
        Initialise the experience buffer
        '''

        self.experience_buffer = np.zeros((self.num_states*self.num_actions, 4), dtype=int)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.experience_buffer[s*self.num_actions+a] = [s, a, 0, s]

        return None

    def _init_history(self):

        '''
        Initialise the history
        '''

        self.history = np.empty((0, 4), dtype=int)

        return None
    
    def _init_action_count(self):

        '''
        Initialise the action count
        '''

        self.action_count = np.zeros((self.num_states, self.num_actions), dtype=int)

        return None
    
    
    def _update_experience_buffer(self, s, a, r, s1):

        '''
        Update the experience buffer (world model)
        Input arguments:
            s  -- initial state
            a  -- chosen action
            r  -- received reward
            s1 -- next state
        '''
        
        # complete the code
        self.experience_buffer[s*self.num_actions+a] = [s,a,r,s1]

        return None

    def _update_qvals(self, s, a, r, s1, bonus=False):

        '''
        Update the Q-value table
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            bonus -- True / False whether to use exploration bonus or not
        '''

        # complete the code
        self.Q[s,a] = self.Q[s,a]+self.alpha*(r+bonus*self.epsilon*np.sqrt(self.action_count[s,a])+self.gamma*np.max(self.Q[s1,:])-self.Q[s,a])


        return None

    def _update_action_count(self, s, a):

        '''
        Update the action count
        Input arguments:
            Input arguments:
            s  -- initial state
            a  -- chosen action
        '''

        # complete the code            
        self.action_count += 1
        self.action_count[s,a] = 0
        
        return None

    def _update_history(self, s, a, r, s1):

        '''
        Update the history
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''

        self.history = np.vstack((self.history, np.array([s, a, r, s1])))

        return None

    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''

        # complete the code
        max_policy_vals = self.Q[s,:]+self.epsilon*np.sqrt(self.action_count[s,:])
        a = np.argmax(max_policy_vals)
        # if given values inside the argmax are the same, it will choose the same thing everytime. 
        # in this case, we should force the random selection of action.
        if len(np.unique(max_policy_vals)) != self.num_actions:
            max_policy_val = np.max(max_policy_vals)
            action_candid = [i for (i,val) in enumerate(max_policy_vals) if val == max_policy_val]
            a = np.random.choice(action_candid)

        return a

    def _plan(self, num_planning_updates):

        '''
        Planning computations
        Input arguments:
            num_planning_updates -- number of planning updates to execute
        '''

        # complete the code
        for _ in range(num_planning_updates):
            
            # choose random state and action
            s = np.random.choice(np.arange(self.num_states))
            a = np.random.choice(np.arange(self.num_actions))
            # next state under the world model
            s,a,r,s1 = self.experience_buffer[s*self.num_actions+a]
            # learning
            self._update_qvals(s, a, r, s1, bonus=True)

        return None

    def get_performace(self):

        '''
        Returns cumulative reward collected prior to each move
        '''

        return np.cumsum(self.history[:, 2])

    def simulate(self, num_trials, reset_agent=True, num_planning_updates=None):

        '''
        Main simulation function
        Input arguments:
            num_trials           -- number of trials (i.e., moves) to simulate
            reset_agent          -- whether to reset all knowledge and begin at the start state
            num_planning_updates -- number of planning updates to execute after every move
        '''

        if reset_agent:
            self._init_q_values()
            self._init_experience_buffer()
            self._init_action_count()
            self._init_history()

            self.s = self.start_state

        for _ in range(num_trials):

            # choose action
            a  = self._policy(self.s)
            # get new state
            s1 = np.random.choice(np.arange(self.num_states), p=self.T[self.s, a, :])
            # receive reward
            r  = self.R[self.s, a]
            # learning
            self._update_qvals(self.s, a, r, s1, bonus=False)
            # update world model 
            self._update_experience_buffer(self.s, a, r, s1)
            # reset action count
            self._update_action_count(self.s, a)
            # update history
            self._update_history(self.s, a, r, s1)
            # plan
            if num_planning_updates is not None:
                self._plan(num_planning_updates)

            if s1 == self.goal_state:
                self.s = self.start_state
            else:
                self.s = s1

        return None
    
class TwoStepAgent(Environment_TwoStepAgent):

    def __init__(self, alpha1, alpha2, beta1, beta2, lam, w, p):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha1 -- learning rate for the first stage \in (0, 1]
            alpha2 -- learning rate for the second stage \in (0, 1]
            beta1  -- inverse temperature for the first stage
            beta2  -- inverse temperature for the second stage
            lam    -- eligibility trace parameter
            w      -- mixing weight for MF vs MB \in [0, 1] 
            p      -- perseveßration strength
        '''

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1  = beta1
        self.beta2  = beta2
        self.lam    = lam
        self.w      = w
        self.p      = p

        return None
    
    def _init_env(self):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''

        Environment_TwoStepAgent.__init__(self)

        return None
    
    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''
        self.Q_MB = np.zeros((self.num_states, self.num_actions))
        self.Q_TD = np.zeros((self.num_states, self.num_actions))
        self.Q_net = np.zeros((self.num_states, self.num_actions))

        return None
    
    def _init_transition_matrix(self):
        '''
        Initialize the transition matrix
        '''
        # hidden transition matrix
        self.trans_mat = np.array([[0.7,0.3],[0.3,0.7]]) # [(s1,a1)->s2,(s1,a1)->s3,(s1,a2)->s2,(s1,a2)->s3]
        # perceived transition matrix - candidates
        self.trans_mat1 = np.array([[0.7,0.3],[0.3,0.7]]) # [(s1,a1)->s2,(s1,a1)->s3,(s1,a2)->s2,(s1,a2)->s3]
        self.trans_mat2 = np.array([[0.3,0.7],[0.7,0.3]]) # [(s1,a1)->s2,(s1,a1)->s3,(s1,a2)->s2,(s1,a2)->s3]
        
        return None
    
    def _init_transition_count(self):
        '''
        Initialize the transition count to decide transition matrix
        '''
        self.trans_count = np.zeros((2,2)) # [a0->s2,a0->s3;a1->s2,a1->s3]
        
        return None
    
    def _init_previous_action(self):
        '''
        Initialize a vatiable to save the previous action
        '''
        # only the top stage action
        self.prev_a1 = -1
          
        return None
    
    def _init_reward_probability(self):
        '''
        Initialize the reward probability
        '''
        self.base_reward_p = np.random.uniform(0.25,0.75,4)
    
    def _reward_prob_generator(self,s2,a2):
        '''
        Set the reward probability from each state, action pair
        
        output: reward probability
        '''
        r_given = np.zeros((self.num_states,self.num_actions))
        r_given[1::,0::] = self.base_reward_p.reshape(2,2)
        
        r_prob = r_given + np.random.normal(loc=0, scale=0.025, size=(self.num_states,self.num_actions))
        
        # compute reflective boundary only for the relevant reward 
        if r_prob[s2,a2] < 0.25:
            r_prob[s2,a2] = (0.25-r_prob[s2,a2])+0.25
        elif r_prob[s2,a2] > 0.75:
            r_prob[s2,a2] = 0.75-(r_prob[s2,a2]-0.75)
        
        return r_prob
    
    def _update_q_td_vals(self,s1,a1,s2,a2,r):
        '''
        Updating Q_TD values
        '''
        self.Q_TD[s1,a1] = self.Q_TD[s1,a1]+self.alpha1*(0+self.Q_TD[s2,a2]-self.Q_TD[s1,a1])
        self.Q_TD[s2,a2] = self.Q_TD[s2,a2]+self.alpha2*(r+0-self.Q_TD[s2,a2])
        self.Q_TD[s1,a1] = self.Q_TD[s1,a1]+self.alpha1*self.lam*(r+0-self.Q_TD[s2,a2])
        
        return None
    
    def _update_q_mb_vals(self):
        '''
        Updating Q_MB values
        '''
        self.Q_MB[1::,:] = self.Q_TD[1::,:] # to update second stages
        # now update the first stage
        if (self.trans_count[0,0]+self.trans_count[1,1]) >= (self.trans_count[1,0]+self.trans_count[0,1]):
            self.Q_MB[0,0] = self.trans_mat1[0,0]*np.max(self.Q_TD[1,:])+self.trans_mat1[0,1]*np.max(self.Q_TD[2,:])
            self.Q_MB[0,1] = self.trans_mat1[1,0]*np.max(self.Q_TD[1,:])+self.trans_mat1[1,1]*np.max(self.Q_TD[2,:])
        else:
            self.Q_MB[0,0] = self.trans_mat2[0,0]*np.max(self.Q_TD[1,:])+self.trans_mat2[0,1]*np.max(self.Q_TD[2,:])
            self.Q_MB[0,1] = self.trans_mat2[1,0]*np.max(self.Q_TD[1,:])+self.trans_mat2[1,1]*np.max(self.Q_TD[2,:])
        
        return None
    
    def _update_q_net_vals(self):
        '''
        Updating Q_net values
        '''
        self.Q_net[1::,:] = self.Q_TD[1::,:]
        self.Q_net[0,:] = self.w*self.Q_MB[0,:]+(1-self.w)*self.Q_TD[0,:]
        
        return None
    
    def _policy(self,s):
        '''
        Policy based on choice matrix
        
        output: action
        '''
        if s == 0:
            P_0 = np.exp(self.beta1*(self.Q_net[s,0]+self.p*(self.prev_a1==0))) 
            P_1 = np.exp(self.beta1*(self.Q_net[s,1]+self.p*(self.prev_a1==1)))
            P_0 = P_0 / (P_0+P_1)
        else:
            P_0 = np.exp(self.beta2*(self.Q_net[s,0])) 
            P_1 = np.exp(self.beta2*(self.Q_net[s,1]))
            P_0 = P_0 / (P_0+P_1)
        
        p_choice = np.array([P_0,1-P_0])
        a = np.random.choice(np.arange(self.num_actions),p=p_choice)
        
        return a
                             
    def _get_next_state(self,s,a):
        '''
        Send the agent to the next state based on action and transition probability
        
        output: next state
        '''
        if s == 0:
            p_trans = self.trans_mat[a,:]
            next_s = np.random.choice([1,2],p=p_trans)
        else:
            next_s = self.start_state
            
        return next_s
        
    def _init_history(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.history = np.empty((0, 3), dtype=int)

        return None
    
    def _update_history(self, a, s1, r1):

        '''
        Update history
        Input arguments:
            a  -- first stage action
            s1 -- second stage state
            r1 -- second stage reward
        '''

        self.history = np.vstack((self.history, [a, s1, r1]))

        return None
    
    def get_stay_probabilities(self):

        '''
        Calculate stay probabilities
        '''

        common_r      = 0
        num_common_r  = 0
        common_nr     = 0
        num_common_nr = 0
        rare_r        = 0
        num_rare_r    = 0
        rare_nr       = 0
        num_rare_nr   = 0

        num_trials = self.history.shape[0]
        for idx_trial in range(num_trials-1):
            a, s1, r1 = self.history[idx_trial, :]
            a_next    = self.history[idx_trial+1, 0]

            # common
            if (a == 0 and s1 == 1) or (a == 1 and s1 == 2):
                # rewarded
                if r1 == 1:
                    if a == a_next:
                        common_r += 1
                    num_common_r += 1
                else:
                    if a == a_next:
                        common_nr += 1
                    num_common_nr += 1
            else:
                if r1 == 1:
                    if a == a_next:
                        rare_r += 1
                    num_rare_r += 1
                else:
                    if a == a_next:
                        rare_nr += 1
                    num_rare_nr += 1
                    

        return np.array([common_r/num_common_r, rare_r/num_rare_r, common_nr/num_common_nr, rare_nr/num_rare_nr])

    def simulate(self, num_trials):

        '''
        Main simulation function
        Input arguments:
            num_trials -- number of trials to simulate
        '''
        self._init_q_values()
        self._init_transition_matrix()
        self._init_transition_count()
        self._init_previous_action()
        self._init_reward_probability()
        self._init_history()
            
        # complete the code
        for _ in range(num_trials):
            # start at the first stage every time
            s1 = self.start_state
            # choose the action at the first stage
            a1 = self._policy(s1)
            self.prev_a1 = a1
            # get the next state
            s2 = self._get_next_state(s1,a1)
            # compute transition count
            self.trans_count[a1,s2-1] += 1 # -1 is to match the dimension
            # choose the action at the second stage
            a2 = self._policy(s2)
            # get the reward prob
            r_prob = self._reward_prob_generator(s2,a2);
            # now compute reward 
            r = np.random.choice([0,1],p=[1-r_prob[s2,a2],r_prob[s2,a2]])
            # update Q_TD
            self._update_q_td_vals(s1,a1,s2,a2,r)
            # update Q_MB
            self._update_q_mb_vals()
            # update Q_net
            self._update_q_net_vals()
            # update the history
            self._update_history(a1, s2, r)
            
        return None