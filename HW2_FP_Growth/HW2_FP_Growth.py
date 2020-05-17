# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:24:34 2020

@author: g1022
"""
import sys

# set arguments
min_support = sys.argv[1]
min_support = float(min_support)
input_data = sys.argv[2]
output_file = sys.argv[3]

# load in dataset
with open(str(input_data), 'r') as file:
    data = file.read().splitlines()
    
data = [list(map(int, i.split(','))) for i in data]
N = len(data)

def create_dataset(dataset):
    dat_dict = {}
    for i in dataset:
        key = frozenset(i)
        if key in dat_dict.keys():
            dat_dict[frozenset(i)] += 1
        else:
            dat_dict[frozenset(i)] = 1
        
    return dat_dict

# define a tree class
class treeNode:
    
    def __init__(self, value_name, freq_count, parent_node):
        self.name = value_name
        self.count = freq_count
        self.node_link = None
        self.parent = parent_node
        self.children = {}

    def add_count(self, freq_count):
        self.count += freq_count
        
# construct FP tree
# add element to the node link
def update_header(node_list, target):
    while (node_list.node_link != None):
        node_list = node_list.node_link
        
    node_list.node_link = target

# update FP tree by the frequency item
def update_tree(items, input_tree, header_table, count):
    if items[0] in input_tree.children:
        input_tree.children[items[0]].add_count(count)
    else:
        input_tree.children[items[0]] = treeNode(items[0], count, input_tree)
        
        if header_table[items[0]][1] == None:
            header_table[items[0]][1] = input_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], input_tree.children[items[0]])

    if len(items) > 1:
        update_tree(items[1::], input_tree.children[items[0]], header_table, count)

# create FP tree
def create_tree(dataset, min_support):
    header_table = {}
    for i in dataset:
        for item in i:
            header_table[item] = header_table.get(item, 0) + dataset[i]
            
    # remove support < min_support
    keys = list(header_table.keys())
    for k in keys:
        if header_table[k]/N < min_support:
            del(header_table[k])
            
    freq_item = set(header_table.keys())
    if len(freq_item) == 0:
        return None, None
    
    for k in header_table:
        header_table[k] = [header_table[k], None]
    output_tree = treeNode('Null Set', 1, None)
    
    # traverse dataset again
    for k, v in dataset.items():
        one_item_dict = {}
        
        for item in k:
            if item in freq_item:
                one_item_dict[item] = header_table[item][0]
                
        if len(one_item_dict) > 0:
            ordered_items = [v[0] for v in sorted(one_item_dict.items(), key=lambda p: (p[1], p[0]), reverse=True)]
            update_tree(ordered_items, output_tree, header_table, v)
            
    return output_tree, header_table

def pre_tree(leaf_node, prefix_path):
    if leaf_node.parent != None:
        prefix_path.append(leaf_node.name)
        pre_tree(leaf_node.parent, prefix_path)

def find_prefix_path(base_pattern, treeNode):
    conditional_patterns = {}
    while treeNode != None:
        prefix_path = []
        pre_tree(treeNode, prefix_path)
        
        if len(prefix_path) > 1:
            conditional_patterns[frozenset(prefix_path[1:])] = treeNode.count
            
        treeNode = treeNode.node_link
        
    return conditional_patterns

def mining_tree(input_tree, header_table, min_support, preFix, freq_item_list, freq_items_support):
    sorted_header_table = sorted(header_table.items(), key=lambda p: p[1][0])
    high_freq = [v[0] for v in sorted_header_table]
    for base_pattern in high_freq:
        new_freq_set = preFix.copy()
        new_freq_set.add(base_pattern)
        freq_item_list.append(list(new_freq_set))
        base_conditional_pattern = find_prefix_path(base_pattern, header_table[base_pattern][1])
        conditional_tree, new_header_table = create_tree(base_conditional_pattern, min_support)

        if new_header_table != None:
            for k,v in new_header_table.items():
                new_key = list(new_freq_set) + [k]
                new_key.sort()
                freq_items_support.append([new_key, "{:.4f}".format(v[0]/N)])
                
            mining_tree(conditional_tree, new_header_table, min_support, new_freq_set, freq_item_list, freq_items_support)
            
# mine tree and get frequency items and support
init_set = create_dataset(data)
FP_tree, header_table_final = create_tree(init_set, min_support)

freq_items = []
freq_items_support = []
mining_tree(FP_tree, header_table_final, min_support, set([]), freq_items, freq_items_support)

# arrange data to final form
freq_items_dict = dict()
for element in freq_items: 
    if len(element) not in freq_items_dict: 
        freq_items_dict[len(element)] = [element] 
    else:
        freq_items_dict[len(element)] += [element] 
        
for i in freq_items_dict[1]:
    freq_items_support.append([[i[0]], "{:.4f}".format(header_table_final[i[0]][0]/N)])
    
freq_items_support_dict = dict()
for element in freq_items_support: 
    if len(element[0]) not in freq_items_support_dict: 
        freq_items_support_dict[len(element[0])] = [element] 
    else:
        freq_items_support_dict[len(element[0])] += [element] 
        
result = []
for key in sorted(freq_items_support_dict):
    freq_items_support_dict[key].sort(key=lambda x: (x[0]))
    result.append(freq_items_support_dict[key]) 

flat_list = [item for sublist in result for item in sublist]
final_output = [(','.join(str(i) for i in x[0])+':'+x[1]) for x in flat_list]

with open(str(output_file), 'w') as output:
    for row in final_output:
        output.write(str(row) + '\n')
