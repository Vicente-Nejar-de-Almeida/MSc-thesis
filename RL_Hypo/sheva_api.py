import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool
from numpyencoder import NumpyEncoder
import random
import math
from collections import deque, namedtuple
import itertools
import sys
import os
import time
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = SCRIPT_DIR+'/../Group_testing/'

sys.path.append(os.path.dirname(SCRIPT_DIR))

from source_code.models.models import get_data
from source_code.core.pivot_handler import PeriodHandler
from source_code.core.group_handler import GroupHandler
from source_code.utils.tools import name_groups
from source_code.core.hypothesis_evaluation.test_handler_2 import test_groups
from config import THRESHOLD_INDEPENDENT_TEST

torch.autograd.set_detect_anomaly(True)

class Agent(nn.Module):
    def __init__(self, state_size, hidden_size, num_groups_per_step, num_attr, num_agg):

        super(Agent, self).__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size

        self.num_groups_per_step = num_groups_per_step
        self.num_agg = num_agg
        self.num_attr = num_attr

        self.fc_1 = nn.Linear(state_size, hidden_size).double()
        #torch.nn.init.xavier_uniform_(self.fc_1.weight)
        self.relu = nn.ReLU()

        self.fc_2 = nn.Linear(hidden_size, hidden_size).double()
        #torch.nn.init.xavier_uniform_(self.fc_2.weight)
        self.relu2 = nn.ReLU()

        self.fc_3 = nn.Linear(hidden_size, num_groups_per_step*num_attr).double()
        #torch.nn.init.xavier_uniform_(self.fc_3.weight)
        
        
        self.fc_agg = nn.Linear(state_size, hidden_size).double()
        #torch.nn.init.xavier_uniform_(self.fc_agg.weight)
        self.relu_agg = nn.ReLU()

        self.fc_agg_2 = nn.Linear(hidden_size, num_agg).double()
        #torch.nn.init.xavier_uniform_(self.fc_agg_2.weight)

    def forward(self, input_state):
        out = self.fc_1(input_state.clone())
        out = self.relu(out)

        out_2 = self.fc_2(out)
        out_2 = self.relu2(out_2)

        out_agg = self.fc_agg(input_state.clone())
        out_agg = self.relu_agg(out_agg)

        return self.fc_3(out_2), self.fc_agg_2(out_agg)

class Agent_explore_exploit(nn.Module):
    def __init__(self, state_size, hidden_size):

        super(Agent_explore_exploit, self).__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size

        self.fc_1 = nn.Linear(state_size, hidden_size).double()
        #torch.nn.init.xavier_uniform_(self.fc_1.weight)
        self.relu = nn.ReLU()

        self.fc_2 = nn.Linear(hidden_size, hidden_size).double()
        #torch.nn.init.xavier_uniform_(self.fc_2.weight)
        self.relu2 = nn.ReLU()

        self.fc_3 = nn.Linear(hidden_size, 2).double()
        #torch.nn.init.xavier_uniform_(self.fc_3.weight)

    def forward(self, input_state):
        out = self.fc_1(input_state.clone())
        out = self.relu(out)

        out_2 = self.fc_2(out)
        out_2 = self.relu2(out_2)

        return self.fc_3(out_2)


class GetRegions(object):
    def __init__(self, policies, dataset, num_attr=6, num_groups_per_step=4, num_agg=3):
        if dataset == 'MovieLens':
            self.support = 5
            self.group_vector_size = 69
            suffix = ''
            self.vv = 2.5
        elif dataset == 'TestAssignment':
            self.support = 1
            self.group_vector_size = 471
            suffix = '_assignment'
            self.vv = 0.5

        self.__prepare_data(dataset)

        self.num_agg = num_agg
        self.num_groups_per_step = num_groups_per_step
        self.hidden_size = 128
        self.state_size = self.num_groups_per_step*self.group_vector_size+self.num_agg
        
        self.policies = {}
        self.expl_policies = {}

        for policy in policies:

            policy_net = Agent(self.state_size, self.hidden_size, self.num_groups_per_step, self.num_attr, self.num_agg)
            policy_net.load_state_dict(torch.load(f'policy_net_{policy}{suffix}.pth'))
            policy_net.eval()

            policy_net_explore_exploit = Agent_explore_exploit(self.state_size, self.hidden_size)
            policy_net_explore_exploit.load_state_dict(torch.load(f'policy_net_explore_exploit_{policy}{suffix}.pth'))
            policy_net_explore_exploit.eval()

            self.policies[policy] = policy_net
            self.expl_policies[policy] = policy_net_explore_exploit

        self.idx2agg = {0:'mean', 1:'variance', 2:'distribution'}
        self.agg2idx = {'mean':np.array([1,0,0]), 'variance':np.array([0,1,0]), 'distribution':np.array([2,0,1])}

        self.idx2method = {-1:'TRAD_BY', 0:'TRAD_BN',1:'COVER_G',2:'coverage_Side_1',3:'coverage_Side_2', 4:'COVER_⍺',\
        5:'β-Farsighted',6:'γ-Fixed',7:'ẟ-Hopeful',8:'Ɛ-Hybrid',9:'Ψ-Support'}

        self.ground_truth_dict = pickle.load( open(f'ground_truth{suffix}.pickle','rb') )
        self.ground_truth_dict = eval(self.ground_truth_dict)

    def worker(self, args):
        key1 = args[0]
        key2 = args[1]

        if key1 in self.all_names_2:
            liste = []
            key = list(key1)

            for val in self.column_to_values[key2]:
                a = set( key+[val] )
                if a in self.all_names_2:
                    aa = self.all_names[self.all_names_2.index(a)]
                    liste.append(aa)
            
            name_key = self.all_names[self.all_names_2.index(key1)]

            return (name_key,key2),liste

        return None

    def __name_groups_2(self, df):
        index = df.index.names

        if (isinstance(df,int)) or (isinstance(df,float)) or (isinstance(df,str)):
            return str(df)
        
        df = df.drop(columns=['article_id','rating','cust_id'])

        #columns = [col for col in df.columns if len(df[col].unique())==1]
        columns = list(index)
        columns.sort()

        return ['_'.join(i) for i in df.reset_index()[columns].drop_duplicates().values][0]

    def __prepare_data(self, dataset):
        group_handler = GroupHandler()
        period_handler = PeriodHandler()

        if dataset == 'MovieLens':
            df = pd.read_csv(f'MovieLens.csv')
            df.index = pd.to_datetime(pd.to_datetime(df.timestamp).dt.date)
        elif dataset == 'TestAssignment':
            df = pd.read_csv(f'answers.csv')
            df = df.drop(columns=['id','random','log','answer'])

            df2 = pd.read_csv(f'items.csv')
            df2['operation'] = df2.data.apply(lambda x: 'unique' if 'operation' not in eval(x).keys() else eval(x)['operation'])
            df2['operands'] = df2.data.apply(lambda x: len(eval(x)['operands']) )

            df2 = df2.drop(columns=['skill_lvl_3','data','answer','question'])
            df2 = df2.rename(columns={'id':'item'})
            df2 = df2.rename(columns={'skill':'skill_lvl_3'})
            df = pd.merge(df,df2, on='item')

            df = df.rename(columns={'item':'article_id','student':'cust_id','correct':'rating','time':'timestamp'})
            df = df.drop(columns=['response_time', 'answer_expected'])
            df.at[df.skill_lvl_2.isna(),['skill_lvl_2']] = 2018

            df.timestamp = df.timestamp.apply(lambda x: str(x).split(' ')[0])
            df.timestamp = pd.to_datetime(df.timestamp)
            df.index = pd.to_datetime(df.timestamp.dt.date)

            df.skill_lvl_1 = df.skill_lvl_1.astype('str')
            df.skill_lvl_3 = df.skill_lvl_3.astype('str')

            df.skill_lvl_2 = df.skill_lvl_2.astype('int32')
            df.skill_lvl_2 = df.skill_lvl_2.astype('str')

            df.operands = df.operands.astype('str')

            df.visualization = df.visualization.apply(lambda x: x.replace('_','-'))


        self.users = set(df.cust_id.unique())

        columns = df.columns
        columns = [col for col in columns if col not in ('purchase','transaction_date','timestamp')]

        self.columns_2 = [col for col in columns if col not in ('cust_id','article_id','rating')]

        self.types_columns = dict()
        self.one_hot_columns = dict()

        self.values_to_columns = dict()
        self.column_to_values = dict()

        for col in columns:
            if col in ['article_id','cust_id']:
                continue
            
            df_loc = df[col]

            typ = df_loc.dtype
            self.types_columns[col] = typ

            if col == 'genre':
                l = []
                df_loc.drop_duplicates().apply(lambda x: l.extend( x.split('|') ) )
                all_values = list(set(l))
            else:
                all_values = df_loc.unique()

            if col != 'rating':
                self.column_to_values[col] = list(all_values)

                for val in all_values:
                    self.values_to_columns[val] = col

            if typ == 'object':
                one = nn.functional.one_hot(torch.tensor(range(len(all_values))), num_classes=len(all_values))

                for i,val in enumerate(all_values):
                    self.one_hot_columns[col,val] = one[i,:].numpy()

        hierarchy_groups = []

        for i in range(1,len(self.columns_2)):
            a = list(itertools.combinations(self.columns_2, i))
            for j in a:
                split_ons = set(self.columns_2) - set(j)
                
                vals = [self.column_to_values[m] for m in set(j)]
                vals = list(itertools.product(*vals))

                for m in vals:
                    for k in split_ons:
                        hierarchy_groups.append( [set(m), k] )

        self.idx2columns = {idx:attr for idx,attr in enumerate(self.columns_2)}

        first_date = df.index.min() #Timestamp is the index

        df = df[columns]
        df_2 = period_handler.period(df, 'One-Sample', 'time', [first_date], [0,2])

        groups = None

        if __name__ == '__main__':
            groups = [ group_handler.groups(d, ['']) for i,d in enumerate(df_2) ] #Create all possible groups

        groups = [ [df for df in grp if len(df)>self.support  ] for grp in groups ]

        self.nameGrp_2_index = [ {self.__name_groups_2(df):df.reset_index() for idx,df in enumerate(grp)} for grp in groups ]

        self.all_names = list(self.nameGrp_2_index[0].keys())
        self.all_names_2 = [set(sorted(set(name.split('_')))) for name in self.all_names]

        pool = Pool()      
        res = pool.map(self.worker, hierarchy_groups)
        pool.close()

        res = [a for a in res if a is not None]

        self.hierarchy_groups = dict()

        for a in res:
            self.hierarchy_groups[ a[0] ] = a[1]

        del res

        self.all_names_2 = list(self.all_names)

        self.all_names_2 = [name for name in self.all_names_2 if len(name.split('_'))==1]
        self.all_names_2 = sorted(self.all_names_2, key=lambda x:len(self.nameGrp_2_index[0][x].cust_id.unique()), reverse=True)

        self.num_attr = len(self.columns_2)
        #self.start_cases = []

        #for i in range(1,3):
        #    a = list(itertools.combinations(self.all_names_2, i))
        #    self.start_cases.extend(a)
    
    def __data2state(self, group_state):
        state = np.array([])

        for grp in group_state:
            grp = grp.reset_index()
            for col,typ in self.types_columns.items():

                if typ == 'object':
                    val_counts = grp[[col]].value_counts()
                    rep = self.one_hot_columns[ col,val_counts.index[0][0] ] * val_counts[0]

                    for i in val_counts.index[1:]:
                        rep += self.one_hot_columns[ col,i[0] ] * val_counts[i]

                else:
                    rep = grp[[col]].describe().to_numpy().reshape(1,8)
                    rep = rep[0]

                state = np.concatenate( (state,rep) )

        return state

    def __get_mask(self, group_state_names, filters_attributes):
        mask_action_2 = [set(filters_attributes)-set([self.values_to_columns[val] for val in grp]) for grp in group_state_names]
        mask_action = []

        for i in range(self.num_groups_per_step*self.num_attr):
            grp_idx = i//self.num_attr

            if grp_idx < len(group_state_names):
                idx = i%self.num_attr

                if self.idx2columns[idx] in mask_action_2[grp_idx]:
                    mask_action.append(0)
                else:
                    mask_action.append(-np.Inf)
            else:
                mask_action.append(-np.Inf)

        return mask_action

    def __get_mask_2(self, group_state_names):
        mask_action_2 = [set(self.columns_2)-set([self.values_to_columns[val] for val in grp]) for grp in group_state_names]
        mask_action = []

        for i in range(self.num_groups_per_step*self.num_attr):
            grp_idx = i//self.num_attr

            if grp_idx < len(group_state_names):
                idx = i%self.num_attr

                if self.idx2columns[idx] in mask_action_2[grp_idx]:
                    mask_action.append(0)
                else:
                    mask_action.append(-np.Inf)
            else:
                mask_action.append(-np.Inf)

        return mask_action

    def __name_to_attributes(self, name):
        l = [self.values_to_columns[i] for i in name.split('_')]
        return '_'.join(l)

    def get_results(self, policy_name, prev_selected=None, list_regions=None, agg_function=None, filters_functions=None, filters_attributes=None):
        policy = self.policies[policy_name]
        policy_expl = self.expl_policies[policy_name]

        done = True
        first_ite = False

        #Apply a mask of Agg

        if filters_functions is None:
            filters_functions = ['mean','variance','distribution']

        if filters_attributes is None:
            filters_attributes = self.columns_2
        
        mask_agg = [0 if self.idx2agg[i] in filters_functions else -np.Inf for i in range(self.num_agg)]

        if list_regions is None:
            shuffled_cases = list(self.all_names_2)
            random.shuffle(shuffled_cases)
            group_state_names = shuffled_cases[:1]
        else:
            if isinstance(list_regions,str):
                group_state_names = [list_regions]
            else:
                group_state_names = list_regions

        if prev_selected is None:
            first_ite = True
            prev_selected = group_state_names[0]

            names_to_dict = set(sorted(set(group_state_names[0].split('_'))))
            names_to_dict = set([self.values_to_columns[val] for val in names_to_dict])
        else:
            names_to_dict = set(sorted(set(prev_selected.split('_'))))
            names_to_dict = set([self.values_to_columns[val] for val in names_to_dict])

        group_state = [self.nameGrp_2_index[0][i] for i in group_state_names]

        remaining = self.num_groups_per_step - len(group_state)
        zeros = np.zeros(remaining*self.group_vector_size)

        state_groups = self.__data2state(group_state)
        state_groups = np.concatenate( (state_groups,zeros) )

        if agg_function is None:
            agg_function = random.choice(filters_functions)
        
        state_hypo = self.agg2idx[agg_function]

        state = np.concatenate( (state_groups,state_hypo) )
        state = torch.tensor(state).double().view(1,-1)

        action_explore_exploit = policy_expl(state)
        action_explore_exploit = action_explore_exploit.argmax(1).view(-1,1)#torch.distributions.Categorical(logits=action_explore_exploit).sample().view(-1,1)
        action_explore_exploit = action_explore_exploit[0]

        if action_explore_exploit[0]==0 and first_ite == False:
            #Explore
            group_state_names = [name for name in self.all_names_2 if self.values_to_columns[name] not in names_to_dict]
            random.shuffle(group_state_names)
            group_state_names = group_state_names[:self.num_groups_per_step]

            group_state = [self.nameGrp_2_index[0][i] for i in group_state_names]
            state_2 = self.__data2state(group_state)

            remaining_2 = self.num_groups_per_step - len(group_state)
                
            if remaining_2 > 0:
                zeros = np.zeros(remaining_2*self.group_vector_size)
                state_2 = np.concatenate( (state_2,zeros) )

            state_2 = np.concatenate( (state_2,state_hypo) )
            state_2 = torch.tensor(state_2).double().view(1,-1)

            explore = True

        else:
            #Exploit
            state_2 = state.clone().detach()
            explore = False
        
        names_to_dict = [set(sorted(set(name.split('_')))) for name in group_state_names]
        mask = self.__get_mask(names_to_dict, filters_attributes)

        action_group, action_agg = policy(state_2)

        mask = torch.Tensor(mask).float().view(action_group.size())
        action_group = torch.add(action_group, mask)
        action_group = action_group.argmax(1).view(-1,1)
        #action_group = torch.distributions.Categorical(logits=action_group).sample().view(-1,1)

        mask_agg = torch.Tensor(mask_agg).float().view(action_agg.size())
        action_agg = torch.add(action_agg, mask_agg)
        action_agg = action_agg.argmax(1).view(-1,1)
        #action_agg = torch.distributions.Categorical(logits=action_agg).sample().view(-1,1)
        
        action_group = action_group[0]
        action_agg = action_agg[0][0].item()

        input_data_region = action_group[0].item()//self.num_attr
        split_attribute = action_group[0].item()%self.num_attr

        selected_group = group_state[input_data_region]
        selected_group_name = group_state_names[input_data_region]

        split_attribute = self.idx2columns[split_attribute]

        agg_type = self.idx2agg[ action_agg ]
        new_state_hypo = self.agg2idx[ agg_type ]

        test_arg = ['One-Sample']

        if action_agg < 2:
            test_arg.append(self.vv)
        else:
            test_arg.append('uniform')

        if len(selected_group_name.split('_'))+1 == len(self.columns_2):
            done = False

        top_n = [self.num_groups_per_step]
        num_hyps = [1]
        approaches = [-1] #Alpha investing
        alpha = 0.05
        dimension = 'rating'

        users = set(selected_group.cust_id.unique())

        stats, results, names = test_groups([selected_group],[selected_group_name], split_attribute, None, self.nameGrp_2_index, self.hierarchy_groups, dimension,\
        top_n, num_hyps, approaches, agg_type, test_arg, users, self.support, alpha, verbose=False)

        names = names[0]

        if len(results) == 0:
            group_state_names = []
            groups_results = []

            min_p_val = -1
            max_p_val = -1
            sum_p_val = -1
            cov_total = 0

            fdr = 0
            power = 0

            done = False
        else:
            groups_results = results[num_hyps[0]*100][self.num_groups_per_step][self.idx2method[approaches[0]]][0]
            #ground_truth = results[num_hyps[0]*100][self.num_groups_per_step][self.idx2method[approaches[1]]][0]

            p_values = groups_results['p-value'].values

            groups_results = groups_results.group1.unique()
            groups_results = list(groups_results)

            #ground_truth = ground_truth.group1.unique()
            #ground_truth = list(ground_truth)

            ground_truth = self.ground_truth_dict[agg_type]

            if len(ground_truth) != 0:
                power = len( set(groups_results).intersection(set(ground_truth)) )/len( set(ground_truth) )
            else:
                power = 0

            group_state_names = groups_results
        
            if len(group_state_names) == self.num_groups_per_step:
                stats = stats[num_hyps[0]*100][self.num_groups_per_step][0]

                min_p_val = stats.Min_p_value_BY.values[0]
                max_p_val = stats.Max_p_value_BY.values[0]
                sum_p_val = stats.Sum_p_value_BY.values[0]
                cov_total = stats.Cov_total_BY.values[0]

                fdr = len( set(group_state_names)-set(ground_truth) )/len( set(group_state_names) )

            elif len(group_state_names) != 0:
                stats = stats[num_hyps[0]*100][self.num_groups_per_step][0]

                min_p_val = stats.Min_p_value_BY.values[0]
                max_p_val = stats.Max_p_value_BY.values[0]
                sum_p_val = stats.Sum_p_value_BY.values[0]
                cov_total = stats.Cov_total_BY.values[0]

                fdr = len( set(group_state_names)-set(ground_truth) )/len( set(group_state_names) )

            else:

                min_p_val = -1
                max_p_val = -1
                sum_p_val = -1
                cov_total = 0
                
                fdr = 0

                done = False
        
        attribute_data_region = self.__name_to_attributes(selected_group_name)
        users_data_regions = set(selected_group.cust_id.unique())
        hypotheses = test_arg+[agg_type]
        sizes = [len(self.nameGrp_2_index[0][i]) for i in group_state_names]

        if explore:
            explor_exploit = 'Explore'
        else:
            explor_exploit = 'Exploit'

        attribute_set_output_data_regions = [self.__name_to_attributes(i) for i in group_state_names]
        users_set_output_data_regions = [set(self.nameGrp_2_index[0][i].cust_id.unique()) for i in group_state_names]

        dic={'power':[power],
        'fdr':[fdr],
        'max_pval':[max_p_val],
        'min_pval':[min_p_val],
        'sum_pval':[sum_p_val],
        'coverage':[cov_total],
        'firs_data_region':[prev_selected],
        'input_data_region':[selected_group_name],
        'attributes_combination_input_data_region':[attribute_data_region],
        'cust_ids_input_data_region':[users_data_regions],
        'hypothesis':[hypotheses],
        'action':[explor_exploit],
        'size_output_set':[len(group_state_names)],
        'output_data_regions':[group_state_names],
        'attributes_combination_output_data_regions':[attribute_set_output_data_regions],
        'cust_ids_output_data_regions':[users_set_output_data_regions],
        'size_ouptput_data_regions':[sizes],
        'done':[not done]}

        results = pd.DataFrame(data=dic)

        return results


def getregions_hypo2(current_iteration,originaldf_all):
      originaldf_all_sample=originaldf_all

      atributes_REGION=str(current_iteration.attributes_combination_input_data_region)
      features_REGION=str(current_iteration.input_data_region)


      atributes_REGION=atributes_REGION.translate({ord('\''): None})
      atributes_REGION=atributes_REGION.translate({ord('\"'): None})
      atributes_REGION=atributes_REGION.translate({ord('['): None})
      atributes_REGION=atributes_REGION.translate({ord(']'): None})
      atributes_REGION=atributes_REGION.translate({ord(' '): None})
      features_REGION=features_REGION.translate({ord('\''): None})
      features_REGION=features_REGION.translate({ord('\"'): None})
      features_REGION=features_REGION.translate({ord('['): None})
      features_REGION=features_REGION.translate({ord(']'): None})
      features_REGION=features_REGION.translate({ord(' '): None})
      atributes_REGION = atributes_REGION.split('_')
      features_REGION = features_REGION.split('_')
      #originaldf_all_sample=originaldf_all.sample(n = 20000)
      RegionD=originaldf_all.copy()
      for i in range(len(atributes_REGION)):
            #print(atributes_REGION[i])
            try:
                  #if (atributes_REGION[i]=="genre"):
                  #print(features_REGION[i])
                  RegionD=RegionD[RegionD[atributes_REGION[i]].str.contains(features_REGION[i])]
                  #print(RegionD)
                  #else:
                  #RegionD=RegionD[(RegionD[atributes_REGION[i]] == features_REGION[i])]
                  #print('Tamanho da Regiao nas features')
                  #print(len(SubRegion))
                  #print((SubRegion))
            except:
                #SubRegion=[]
                pass

      import pandas as pd
      from sklearn.preprocessing import LabelEncoder




      from sklearn import preprocessing
      #RegionD
      RegionD = RegionD.drop(columns=['tsne-2d-one','tsne-2d-two',"Unnamed: 0","Unnamed: 0.1"])



      sample_df_selected = RegionD.copy()  
      atributes_all=str(current_iteration.attributes_combination_output_data_regions)
      features_all=str(current_iteration.output_data_regions)
      atributes_all = atributes_all.split(',')
      features_all = features_all.split(',')
      users_pol_all=str(current_iteration.cust_ids_output_data_regions)            
      users_pol_all=users_pol_all.split('},')
      dataframe_collection = {}
      for regions_D in range(len(features_all)):
          #print(regions_D)
          atributes = atributes_all[regions_D]
          features = features_all[regions_D]
          atributes=atributes.translate({ord('\''): None})
          atributes=atributes.translate({ord('\"'): None})
          atributes=atributes.translate({ord('['): None})
          atributes=atributes.translate({ord(']'): None})
          atributes=atributes.translate({ord(' '): None})
          features=features.translate({ord('\''): None})
          features=features.translate({ord('\"'): None})
          features=features.translate({ord('['): None})
          features=features.translate({ord(']'): None})
          features=features.translate({ord(' '): None})
          atributes = atributes.split('_')
          features = features.split('_')
          #print(atributes)
          #print(features)
          SubRegion=RegionD
          for i in range(len(atributes)):
            try:
                #if (atributes[i]=="genre"):
                SubRegion=SubRegion[SubRegion[atributes[i]].str.contains(features[i])]
                #else:
                #SubRegion=SubRegion[(SubRegion[atributes[i]] == features[i])]
                #print('Tamanho da Regiao nas features')
                #print(len(SubRegion))
                #print((SubRegion))
            except:
                #SubRegion=[]
                pass
          #print(len(SubRegion))
          users_pol = users_pol_all[regions_D]
          users_pol=users_pol.translate({ord('\''): None})
          users_pol=users_pol.translate({ord('\"'): None})
          users_pol=users_pol.translate({ord('{'): None})
          users_pol=users_pol.translate({ord('}'): None})
          users_pol=users_pol.translate({ord('['): None})
          users_pol=users_pol.translate({ord(']'): None})

          users_pol = users_pol.split(',')
          SubRegion_aux=SubRegion.copy();
          SubRegionUsers=pd.DataFrame()
          SubRegion=pd.DataFrame()
          for i in range(len(users_pol)):
            try:
              users_pol[i]=int(users_pol[i])
              try:
                  SubRegion=SubRegion_aux[(SubRegion_aux["cust_id"] == users_pol[i])]
                  
                  #print(len(SubRegion))
                  SubRegionUsers=SubRegionUsers.append(SubRegion)
                  #print(SubRegion)
              except:
                  SubRegion=[]
                  pass
            except:
              pass
          dataframe_collection[regions_D] = pd.DataFrame(SubRegionUsers)


      aux_i=0
      for i in range(11,len(RegionD.columns)):
        aux_i=aux_i+1

      #RegionD

      atributes_REGION_stats=str(current_iteration.attributes_combination_input_data_region)
      atributes_REGION_stats=atributes_REGION_stats.translate({ord('\''): None})
      atributes_REGION_stats=atributes_REGION_stats.translate({ord('\"'): None})
      atributes_REGION_stats=atributes_REGION_stats.translate({ord('['): None})
      atributes_REGION_stats=atributes_REGION_stats.translate({ord(']'): None})
      atributes_REGION_stats=atributes_REGION_stats.translate({ord(' '): None})
      atributes_REGION_stats = atributes_REGION_stats.split('_')

      #atributes_REGION_stats

      #STARTING TO BUILD THE STATISTICS

      #dataframe_collection
      #RegionD
      #Regionall_statistics=RegionD.iloc[: , [2,3,4,5,6,7,8,9,10]].copy()
      Regionall_statistics=RegionD.copy()

      Regionall_statistics = Regionall_statistics.assign(Region = 0)

      
      for i in range(len(dataframe_collection)):
          Regionall_statistics_aux=dataframe_collection[i].copy()
          try:
            #Regionall_statistics_aux=Regionall_statistics_aux.drop(columns=['cust_id','article_id'])
            pass
          except:
            pass
          Regionall_statistics_aux['Region']=i+1
          Regionall_statistics=Regionall_statistics.append(Regionall_statistics_aux)

      Regionall_statistics=Regionall_statistics.drop(columns=['purchase'])
      #print("AFTER ALL 3")
      #print(Regionall_statistics.columns)
      def Encoder(originaldf_all):
                    columnsToEncode = list(originaldf_all.select_dtypes(include=['category','object']))
                    le = LabelEncoder()
                    for feature in columnsToEncode:
                        try:
                            originaldf_all[feature] = le.fit_transform(originaldf_all[feature])
                        except:
                            pass
                    return originaldf_all



      return(Regionall_statistics)


policies = ['Sig','Cov','Nov','SigCov','SigNov','CovNov','SigCovNov']
originaldf_all=pd.read_csv('originaldf_all.csv', encoding='utf-8')
done = False

# Call the class before starting the loop
get_regions_movie_lens = GetRegions(policies, 'MovieLens')
get_regions_test_assignment = GetRegions(policies, 'TestAssignment')

fakedata = {
                    'power':                                                            1.0,
                    'fdr':                                                                          0.0,
                    'max_pval':                                                                                     0.0,
                    'min_pval':                                                                                     0.0,
                    'sum_pval':                                                                                     0.0,
                    'coverage':                                                                               0.004967,
                    'firs_data_region':                                                                      "Animation",
                    'input_data_region':                                                                      "Animation",
                    'attributes_combination_input_data_region':                                                  "genre",
                    'cust_ids_input_data_region':                     [{6016, 6017, 6018, 6019, 6021, 6022, 6023}],
                    'hypothesis':                                                           [['One-Sample', 2.5, 'mean']],
                    'action':                                                                                  "Exploit",
                    'size_output_set':                                                                                1,
                    'output_data_regions':                                               [['F_Animation']],
                    'attributes_combination_output_data_regions':                     [['gender_genre']],
                    'cust_ids_output_data_regions':                   [[{6017, 6025, 6029, 6031, 6035, 6036, 6037}]],
                    'size_ouptput_data_regions':                                                               [[7]],
                    'done':                                                                                      False}

fakedata2 = {
                    'power':                                                            1.0,
                    'fdr':                                                                          0.0,
                    'max_pval':                                                                                     0.0,
                    'min_pval':                                                                                     0.0,
                    'sum_pval':                                                                                     0.0,
                    'coverage':                                                                               0.004967,
                    'firs_data_region':                                                                      "Action",
                    'input_data_region':                                                                      "Action",
                    'attributes_combination_input_data_region':                                                  "genre",
                    'cust_ids_input_data_region':                     [{6016, 6017, 6018, 6019, 6021, 6022, 6023}],
                    'hypothesis':                                                           [['One-Sample', 2.5, 'mean']],
                    'action':                                                                                  "Explore",
                    'size_output_set':                                                                                1,
                    'output_data_regions':                                               [['F_Action']],
                    'attributes_combination_output_data_regions':                     [['gender_genre']],
                    'cust_ids_output_data_regions':                   [[{6017, 6025, 6029, 6031, 6035, 6036, 6037}]],
                    'size_ouptput_data_regions':                                                               [[7]],
                    'done':                                                                                      False}


def get_htpothesis(attributes_combination_input_data_region, input_data_region, hypothesis):

    atributes_REGION=attributes_combination_input_data_region	
    features_REGION=input_data_region

    """
    atributes_REGION=atributes_REGION.translate({ord('\''): None})
    atributes_REGION=atributes_REGION.translate({ord('\"'): None})
    atributes_REGION=atributes_REGION.translate({ord('['): None})
    atributes_REGION=atributes_REGION.translate({ord(']'): None})
    atributes_REGION=atributes_REGION.translate({ord(' '): None})
    features_REGION=features_REGION.translate({ord('\''): None})
    features_REGION=features_REGION.translate({ord('\"'): None})
    features_REGION=features_REGION.translate({ord('['): None})
    features_REGION=features_REGION.translate({ord(']'): None})
    features_REGION=features_REGION.translate({ord(' '): None})
    """
    atributes_REGION = atributes_REGION.split('_')
    features_REGION = features_REGION.split('_')

    string=""

    values_hyp=hypothesis
    """
    values_hyp=values_hyp.translate({ord('['): None})
    values_hyp=values_hyp.translate({ord(']'): None})
    values_hyp=values_hyp.translate({ord('\''): None})
    values_hyp=values_hyp.translate({ord(' '): None})
    values_hyp=values_hyp.split(',')
    """
    values_hyp[1] = str(values_hyp[1])
    if(values_hyp[2]=="distribution"):
      string=("Groups of users of ")
      for i in range(len(features_REGION)):
        if(atributes_REGION[i]=="age" or atributes_REGION[i]=="gender" or atributes_REGION[i]=="occupation"):
          string=string+atributes_REGION[i]+" = "+features_REGION[i] + " and "
      string=string[0:len(string)-4]
      string=string + " whose rating distribution for movies of "
      for i in range(len(features_REGION)):
        if(atributes_REGION[i]=="genre" or atributes_REGION[i]=="runtimeMinutes" or atributes_REGION[i]=="year"):
          string=string+atributes_REGION[i]+" = "+features_REGION[i] + " and "
      string=string[0:len(string)-4]
      string=string + " does not follow a uniform distribution."
    if(values_hyp[2]=="mean"):
      string=("Groups of users of ")
      for i in range(len(features_REGION)):
        if(atributes_REGION[i]=="age" or atributes_REGION[i]=="gender" or atributes_REGION[i]=="occupation"):
          string=string+atributes_REGION[i]+" = "+features_REGION[i] + " and "
      string=string[0:len(string)-4]
      string=string + " whose rating mean for movies of "
      for i in range(len(features_REGION)):
        if(atributes_REGION[i]=="genre" or atributes_REGION[i]=="runtimeMinutes" or atributes_REGION[i]=="year"):
          string=string+atributes_REGION[i]+" = "+features_REGION[i] + " and "
      string=string[0:len(string)-4]
      string=string + " is greater than " + values_hyp[1] + "."
    if(values_hyp[2]=="variance"):   
      string=("Groups of users of ")
      for i in range(len(features_REGION)):
        if(atributes_REGION[i]=="age" or atributes_REGION[i]=="gender" or atributes_REGION[i]=="occupation"):
          string=string+atributes_REGION[i]+" = "+features_REGION[i] + " and "
      string=string[0:len(string)-4]
      string=string + " whose rating variance for movies of "
      for i in range(len(features_REGION)):
        if(atributes_REGION[i]=="genre" or atributes_REGION[i]=="runtimeMinutes" or atributes_REGION[i]=="year"):
          string=string+atributes_REGION[i]+" = "+features_REGION[i] + " and "
      string=string[0:len(string)-4]
      string=string + " is greater than " + values_hyp[1] + "."

    return string.replace('  ', ' ')


def hypothesis_to_natural_language(df_current_pipeline, dataset='MovieLens'):
    if dataset == 'MovieLens':
        hypothesis = df_current_pipeline['hypothesis'][0]
        input_data_region = df_current_pipeline['input_data_region'][0]
        attributes_combination_input_data_region = df_current_pipeline['attributes_combination_input_data_region'][0]
        natural_language_hypothesis = ''
        return get_htpothesis(attributes_combination_input_data_region, input_data_region, hypothesis)
    else:
        hypothesis = df_current_pipeline['hypothesis'][0]
        return str(hypothesis)


def current_pipeline_to_information(df_current_pipeline, df_current_operator_results, explore_required, dataset='MovieLens'):
    if explore_required:
        action = 'Explore'
    else:
        action = df_current_pipeline['action'][0]
    
    if dataset == 'MovieLens':
        return {
            'name': hypothesis_to_natural_language(df_current_pipeline, dataset),
            'size_output_set': int(df_current_pipeline['size_output_set'][0]),
            'fdr': round(df_current_pipeline['fdr'][0], 4),
            'power': round(df_current_pipeline['power'][0], 4),
            'coverage': round(df_current_pipeline['coverage'][0], 4),
            'user_count': int(df_current_operator_results[df_current_operator_results['Region'] == 0]['cust_id'].nunique()),
            'movie_count': int(df_current_operator_results[df_current_operator_results['Region'] == 0]['article_id'].nunique()),
            'action': action,
            'input_data_region': df_current_pipeline['input_data_region'][0],
            'output_data_regions': df_current_pipeline['output_data_regions'][0],
            'aggregation': df_current_pipeline['hypothesis'][0][2],
        }
    else:
        return {
            'name': hypothesis_to_natural_language(df_current_pipeline, dataset),
            'size_output_set': int(df_current_pipeline['size_output_set'][0]),
            'fdr': round(df_current_pipeline['fdr'][0], 4),
            'power': round(df_current_pipeline['power'][0], 4),
            'coverage': round(df_current_pipeline['coverage'][0], 4),
            'action': action,
            'input_data_region': df_current_pipeline['input_data_region'][0],
            'output_data_regions': df_current_pipeline['output_data_regions'][0],
            'aggregation': df_current_pipeline['hypothesis'][0][2],
        }


def current_operator_results_to_information(df_current_pipeline, df_current_operator_results):
    results = []

    # print(df_current_pipeline['size_output_set'][0], df_current_pipeline['output_data_regions'][0])
    # print(df_current_operator_results['Region'].unique())
    # print(df_current_operator_results)
    for region in df_current_operator_results['Region'].unique():
        # print('Region', region)
        
        if region == 0:
            data_region = df_current_pipeline['input_data_region'][0]
            attributes_data_region = df_current_pipeline['attributes_combination_input_data_region'][0]
        else:
            data_region = df_current_pipeline['output_data_regions'][0][region-1]
            attributes_data_region = df_current_pipeline['attributes_combination_output_data_regions'][0][region-1]

        current_region_df = df_current_operator_results[df_current_operator_results['Region'] == region].copy()
        # print(current_region_df.values.tolist())

        current_result = {}

        # print(current_region_df.columns)
        # current_result['df_data'] = current_region_df.values.tolist()

        current_result['data_region'] = data_region

        attribute_names = attributes_data_region.split('_')
        attribute_values = data_region.split('_')
        breadcrumbs = []
        for name, value in zip(attribute_names, attribute_values):
            breadcrumbs.append(name + ' = ' + value.capitalize())
        # current_result['breadcrumbs'] = breadcrumbs
        current_result['breadcrumbs'] = list(reversed(breadcrumbs))

        current_result['heading_id'] = 'Region' + str(region)
        current_result['collapse_id'] = 'Collapse' + str(region)

        if region == 0:
            current_result['name'] = hypothesis_to_natural_language(df_current_pipeline)
        else:
            current_result['name'] = get_htpothesis(attributes_data_region,
                                                    data_region,
                                                    df_current_pipeline['hypothesis'][0])
        
        current_result['rating_count'] = [len(current_region_df[current_region_df['rating'] == i]) for i in range(1, 6)]
        
        # ['cust_id', 'article_id', 'rating', 'transaction_date', 'gender', 'age', 'occupation', 'genre', 'runtimeMinutes', 'year', 'Region']
        
        current_result['genders'] = [len(current_region_df[current_region_df['gender'] == 'M']), len(current_region_df[current_region_df['gender'] == 'F'])]
        
        current_result['ages'] = [len(current_region_df[current_region_df['age'] == '<18']),
                                  len(current_region_df[current_region_df['age'] == '18-24']),
                                  len(current_region_df[current_region_df['age'] == '25-34']),
                                  len(current_region_df[current_region_df['age'] == '35-44']),
                                  len(current_region_df[current_region_df['age'] == '45-49']),
                                  len(current_region_df[current_region_df['age'] == '50-55']),
                                  len(current_region_df[current_region_df['age'] == '>56'])]
        
        occupations = ["academic-educator", "artist", "clerical-admin", "college-grad student", "customer service", "doctor-health care",
                       "executive-managerial", "farmer", "homemaker", "K-12 student", "lawyer", "programmer", "retired", "sales-marketing", "scientist",
                       "self-employed", "technician-engineer", "tradesman-craftsman", "unemployed", "writer", "other"]
        
        current_result['occupations'] = [len(current_region_df[current_region_df['occupation'] == o]) for o in occupations]

        genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        current_result['genres'] = [len(current_region_df[current_region_df['genre'].str.contains(g)]) for g in genres]

        if df_current_pipeline['hypothesis'][0][2] == 'mean':
            current_result['selected_hypothesis'] = 1
        elif df_current_pipeline['hypothesis'][0][2] == 'variance':
            current_result['selected_hypothesis'] = 2
        elif df_current_pipeline['hypothesis'][0][2] == 'distribution':
            current_result['selected_hypothesis'] = 3

        results.append(current_result)

    return results


from flask import Flask, request, jsonify, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.json_encoder = NumpyEncoder

@app.route('/', methods=['POST'])
def main():
    selected_group = None
    list_regions = None
    aggregation = None

    if 'dataset' in request.json:
        dataset = request.json['dataset']
    else:
        dataset = 'MovieLens'
    
    if 'selected_group' in request.json:
        selected_group = request.json['selected_group']
    
    if 'list_regions' in request.json:
        list_regions = request.json['list_regions']
    
    if 'aggregation' in request.json:
        aggregation = request.json['aggregation']
    
    if 'policy' in request.json:
        policy = request.json['policy']
    else:
        policy = 'Sig'
    
    print('Aggregation:', aggregation)

    explore_required = False

    if (list_regions is None) or (len(list_regions) == 0):
        selected_group = None
        list_regions = None
        aggregation = None
        explore_required = True
    
    if dataset == 'MovieLens':
        try:
            current_pipeline = get_regions_movie_lens.get_results(policy, prev_selected=selected_group, list_regions=list_regions, agg_function=aggregation)
        except:
            current_pipeline = get_regions_movie_lens.get_results(policy)
            explore_required = True

        current_operator_result = getregions_hypo2(current_pipeline.iloc[0], originaldf_all)

        if int(current_operator_result[current_operator_result['Region'] == 0]['cust_id'].nunique()) == 0:
            current_pipeline = get_regions_movie_lens.get_results(policy)
            current_operator_result = getregions_hypo2(current_pipeline.iloc[0], originaldf_all)
            explore_required = True

        return jsonify({
            'pipeline': current_pipeline_to_information(current_pipeline, current_operator_result, explore_required),
            'operator_results': current_operator_results_to_information(current_pipeline, current_operator_result)
        }), 200
    elif dataset == 'TestAssignment':
        try:
            current_pipeline = get_regions_test_assignment.get_results(policy, prev_selected=selected_group, list_regions=list_regions, agg_function=aggregation)
        except:
            current_pipeline = get_regions_test_assignment.get_results(policy)
            explore_required = True
        
        current_operator_result = None
        
        return jsonify({
            'pipeline': current_pipeline_to_information(current_pipeline, current_operator_result, explore_required, dataset),
            'operator_results': None
        }), 200


if __name__ == "__main__":
    app.run()
