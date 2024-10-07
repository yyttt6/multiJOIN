from typing import Literal, Optional, Generator, Tuple, Dict, List, Set, Any, Union
from collections import defaultdict, deque
import io
import re
import os
import csv
import math
import time
import copy
import string
import random
from datetime import datetime

import torch
from torch.fft import fft, ifft
import numpy
import pandas as pd
from tap import Tap
from tqdm import tqdm

from kwisehash import KWiseHash

import heapq
import sys
import utils
    
def id_bigdata(id, node2batches, topk:int):
    
    huge_dict = {}
    batches = [batch for node,batch in node2batches.items() if node.split(".")[0]==id]

    for batch in zip(*batches):
        for line in zip(*batch):
            if line in huge_dict:
                huge_dict[line] += 1
            else:
                huge_dict[line] = 1
    
    bigdata = dict(heapq.nlargest(topk, huge_dict.items(), key=lambda item: item[1]))
    del huge_dict
    return bigdata

def id_maxdata(id, node2batches):
    
    huge_dict = {}
    batches = [batch for node,batch in node2batches.items() if node.split(".")[0]==id]
    
    for batch in zip(*batches):
        for line in zip(*batch):
            if line in huge_dict:
                huge_dict[line] += 1
            else:
                huge_dict[line] = 1
                
    return max(huge_dict.values())
    
class splited_sketches(object):
    def __init__(self, joined_nodes, num_estimates, args):
        self.joined_nodes_num: int
        self.node2axis: Dict[str, int] = {}
        self.bigdata = {}
        self.smldata: torch.Tensor
        self.node2join_indices :Dict[str, list] = {}
        
        for node_idx,node in enumerate(joined_nodes):
            self.node2axis[node] = node_idx
        # print(self.node2axis)
        self.joined_nodes_num = len(joined_nodes)
        
        self.smldata = torch.zeros(num_estimates, args.bins, dtype=torch.long)

    def erase_bigdata(self, bin_hashes, sign_hashes, query, args):
        
        print(self.bigdata)
        for tup,value in self.bigdata.items():
            value = -value
            bins = 0
            for node,axis in self.node2axis.items():
                
                bins = (bins + bin_hashes[query.node2component[node]](torch.tensor([tup[axis]],dtype=torch.long))) % args.bins
                
                for join_idx in self.node2join_indices[node]:
                        value *= sign_hashes[join_idx](torch.tensor([tup[axis]],dtype=torch.long))
        
            self.smldata.scatter_add_(1, bins, value)
        
                         
    def construct(self, node2batches, bigdata, bin_hashes, sign_hashes, query, args):
        
        batches = []
        all_join_indices = []
        components = []
        for node in self.node2axis.keys():
            batches.append(node2batches[node])
            components.append(query.node2component[node])   
            join_indices = []
            for join_idx, join in enumerate(query.joins):
                left, _, right = join
                if left == node or right == node:
                    join_indices.append(join_idx)    
            self.node2join_indices[node] = join_indices
            
            all_join_indices.append(join_indices)   
            for batch in zip(*batches):             
                signs = 1
                bins = 0
                for attr_values, join_indeces, component in zip(
                    batch, all_join_indices, components
                ):
                    bin_hash = bin_hashes[component]
                    bins = bins + bin_hash(attr_values)                  
                    for join_idx in join_indices:
                        sign_hash = sign_hashes[join_idx]
                        signs = signs * sign_hash(attr_values)
                bins = bins % args.bins
                self.smldata.scatter_add_(1, bins, signs)
        self.bigdata = bigdata
        self.erase_bigdata(bin_hashes, sign_hashes, query, args)
    
    def get_sum(self):
        ans = 0.0
        for value in self.bigdata.values():
            ans += int(value)
        ans = torch.sum(self.smldata, dim=1) + ans
        return ans.to(torch.long)
        
    
def combine_splited_sketches(node, bin_hashes, sign_hashes, visited, query, id2sketch, args):
    id, _ = node.split(".")
    sketch = id2sketch[id]
    visited.add(node)
    
    print("id    "+str(id))
    print("node    "+str(node))
    
    for other_node in query.joined_nodes(id):

        if other_node == node: continue
        visited.add(other_node)
        
        print("other_node    "+str(other_node))
        
        tmp_bigdata = None
        tmp_smldata = 1
        
        bin_hash = bin_hashes[query.node2component[other_node]]
        tmp_join_idxs = []
        
        for joined_node,join_idx in query.joined_with_and_idx(other_node).items():
            
            print("joined_node    "+str(joined_node))

            joined_sketch = combine_splited_sketches(joined_node, bin_hashes, sign_hashes, visited, query, id2sketch, args)

            print("joined_sketch    "+str(joined_sketch))
            
            print(tmp_bigdata)
            if tmp_bigdata == None: 
                tmp_bigdata = joined_sketch.bigdata
                tmp_smldata = joined_sketch.smldata
            else:
                left_set, right_set = (set(tmp_bigdata.keys()), set(joined_sketch.bigdata.keys()))
                
                inter_keys = left_set & right_set
                left_keys = left_set - right_set
                right_keys = right_set - left_set
                
                new_bigdata = {}
                for key in inter_keys:
                    new_bigdata[key] = tmp_bigdata[key] * joined_sketch.bigdata[key]
                    if new_bigdata[key] == 0: del new_bigdata[key]
                for key in left_keys:
                    bins = bin_hash(torch.tensor(key))
                    signs = torch.gather(joined_sketch.smldata, 1, bins) 
                    signs *= sign_hashes[join_idx](torch.tensor(key))
                    new_bigdata[key] = (tmp_bigdata[key] * torch.median(signs)).item()
                    if new_bigdata[key] == 0: del new_bigdata[key]
                for key in right_keys:
                    bins = bin_hash(torch.tensor(key))
                    signs = torch.gather(tmp_smldata, 1, bins)
                    for jid in tmp_join_idxs:
                        signs *= sign_hashes[jid](torch.tensor(key))
                    new_bigdata[key] = (torch.median(signs) * joined_sketch.bigdata[key]).item()
                    if new_bigdata[key] == 0: del new_bigdata[key]
                
                tmp_bigdata = new_bigdata
                
                tmp_smldata *= joined_sketch.smldata

            tmp_join_idxs.append(join_idx)
            print(tmp_bigdata)

        idx = sketch.node2axis[other_node]
        for _node in sketch.node2axis.keys():
            if sketch.node2axis[_node] > idx:
                sketch.node2axis[_node] -= 1
        sketch.node2axis.pop(other_node)
        
        new_bigdata = {}
        for tup,value in sketch.bigdata.items():
            attr = tup[idx]
            new_tup = list(tup)
            new_tup.pop(idx)
            new_tup = tuple(new_tup)
            if attr in tmp_bigdata:
                new_bigdata[new_tup] = (value * tmp_bigdata[attr]).item()
                if new_bigdata[new_tup] == 0: del new_bigdata[new_tup]
                tmp_bigdata.pop(attr)
            else:
                bins = bin_hash(torch.tensor([attr])) % args.bins
                signs = torch.gather(tmp_smldata, 1, bins)
                for join_idx in sketch.node2join_indices[other_node]:
                    signs *= sign_hashes[join_idx](torch.tensor([attr]))
                new_bigdata[new_tup] = (value * torch.median(signs)).item()
                if new_bigdata[new_tup] == 0: del new_bigdata[new_tup]
            
        sketch.bigdata = new_bigdata

        for tup,value in tmp_bigdata.items():
            bins = bin_hash(torch.tensor(tup)) % args.bins
            signs = value
            
            for join_idx in sketch.node2join_indices[other_node]:
                signs *= sign_hashes[join_idx](torch.tensor(tup))
            tmp_smldata.scatter_add_(1,bins,signs)
        sketch.smldata = ifft(fft(tmp_smldata).conj() * fft(sketch.smldata)).real
    
        print("sketch.bigdata    "+str(sketch.bigdata))
        
    bin_hash = bin_hashes[query.node2component[node]]
    tmp_join_idxs = set(sketch.node2join_indices[node])
    print(tmp_join_idxs)
    
    for joined_node,join_idx in query.joined_with_and_idx(node).items():
        
        if joined_node in visited: continue
        
        print("joined_node    "+str(joined_node))
        print(f"join_idx is {join_idx}")

        joined_sketch = combine_splited_sketches(joined_node, bin_hashes, sign_hashes, visited, query, id2sketch, args)

        left_set, right_set = (set(sketch.bigdata.keys()), set(joined_sketch.bigdata.keys()))
        
        inter_keys = left_set & right_set
        left_keys = left_set - right_set
        right_keys = right_set - left_set

        new_bigdata = {}

        for key in inter_keys:
            new_bigdata[key] = sketch.bigdata[key] * joined_sketch.bigdata[key]
            if new_bigdata[key] == 0: del new_bigdata[key]
        for key in left_keys:
            bins = bin_hash(torch.tensor(key))
            signs = torch.gather(joined_sketch.smldata, 1, bins)
            signs *= sign_hashes[join_idx](torch.tensor(key))
            new_bigdata[key] = (sketch.bigdata[key] * torch.median(signs)).item()
            if new_bigdata[key] == 0: del new_bigdata[key]
        for key in right_keys:
            bins = bin_hash(torch.tensor(key))
            signs = torch.gather(sketch.smldata, 1, bins)
            for _join_idx in tmp_join_idxs:
                signs *= sign_hashes[_join_idx](torch.tensor(key))
            new_bigdata[key] = (torch.median(signs) * joined_sketch.bigdata[key]).item()
            if new_bigdata[key] == 0: del new_bigdata[key]
                    
        del sketch.bigdata
        sketch.bigdata = new_bigdata
                
        sketch.smldata *= joined_sketch.smldata
        
        print(tmp_join_idxs)
        tmp_join_idxs.remove(join_idx)
        print(tmp_join_idxs)
        print("sketch.bigdata    "+str(sketch.bigdata))
        
    sketch.smldata = sketch.smldata.to(dtype=torch.long)
    return sketch
            

def experiment():

    args = utils.Arguments(underscores_to_dashes=True).parse_args()

    # Get random ID before setting the seed
    exp_id = utils.random_string()
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{date_str}-multijoin-{exp_id}.csv")
    print("Saving results in:", result_path)

    if args.seed is not None:
        seed(args.seed)
        print("Seed", args.seed)

    print("Reading SQL query...")
    benchmark, query_name = args.query.split("-")
    sql_query = utils.read_sql_query(args.data_dir, benchmark, query_name)
    query = utils.Query(sql_query)
    print(query)
    print("Filters:", query.selects)
    print("Joins:", query.joins)
    print(f"Components ({query.num_components}):", query.node2component)

    print("Loading tables...")
    loading_timer = utils.Timer()
    id2table = utils.load_tables(args.data_dir, benchmark, query)
    
    loading_time = loading_timer.stop()
    print(id2table)

    memory_usage = 0

    init_timer = utils.Timer()

    print("Initialize hash functions...")
    if args.method == "count-conv":
        num_estimates = args.estimates * args.medians

        sign_hashes = []
        for _ in query.joins:
            sign_hash = utils.SignHash(num_estimates, k=4)
            memory_usage += sign_hash.fn.seeds.numel() * 8
            sign_hashes.append(sign_hash)

        bin_hashes = []
        for _ in range(query.num_components):
            bin_hash = utils.BinHash(num_estimates, bins=args.bins, k=2)
            memory_usage += bin_hash.fn.seeds.numel() * 8
            bin_hashes.append(bin_hash)
    elif args.method == "our-method":
        num_estimates = args.estimates * args.medians

        sign_hashes = []
        for _ in query.joins:
            sign_hash = utils.SignHash(num_estimates, k=4)
            memory_usage += sign_hash.fn.seeds.numel() * 8
            sign_hashes.append(sign_hash)

        bin_hashes = []
        for _ in range(query.num_components):
            bin_hash = utils.BinHash(num_estimates, bins=args.bins, k=2)
            memory_usage += bin_hash.fn.seeds.numel() * 8
            bin_hashes.append(bin_hash)
            
    print("Initialize sketches...")
    id2sketch: Dict[str, torch.Tensor] = {}
    
    for id in id2table.keys():

        if args.method == "count-conv":
            id2sketch[id] = torch.zeros(num_estimates, args.bins, dtype=torch.long)
        elif args.method == "our-method":
            joined_nodes = list(query.joined_nodes(id))
            id2sketch[id] = splited_sketches(query.joined_nodes(id), num_estimates, args) 
      
        if args.method == "count-conv":
            memory_usage += id2sketch[id].numel() * 8
        elif args.method == "our-method":
            memory_usage += id2sketch[id].smldata.numel() * 8

    init_time = init_timer.stop()

    stream_timer = utils.Timer()

    print("Applying filters...")
    id2mask = utils.make_selection_filters(id2table, query)
    node2batches = utils.prepare_batches(id2table, id2mask, args.batch_size, query)

    id2maxdata = {id:id_maxdata(id, node2batches) for id in id2table.keys()}
    skew_ids = set([item[0] for item in heapq.nlargest(1, id2maxdata.items(), key = lambda x: x[1])])
    
    if args.method == "count-conv":

        for id in id2table:
            batches = []
            all_join_indices = []
            components = []

            for attr in query.id2joined_attrs[id]:
                node = f"{id}.{attr}"
                batches.append(node2batches[node])
                components.append(query.node2component[node])

                join_indices = []

                for join_idx, join in enumerate(query.joins):
                    left, _, right = join

                    if left == node or right == node:
                        join_indices.append(join_idx)

                all_join_indices.append(join_indices)

            # Add data to sketch

            for batch in zip(*batches):
                #这个batch并行地取所有特征，每个特征取一个tensor
                signs = 1
                bins = 0

                # For each joined attribute of table id
                for attr_values, join_indices, component in zip(
                    batch, all_join_indices, components
                ):
                    #遍历每一个特征
                    bin_hash = bin_hashes[component]
                    bins = bins + bin_hash(attr_values)

                    # For each join with attribute
                    for join_idx in join_indices:

                        sign_hash = sign_hashes[join_idx]
                        signs = signs * sign_hash(attr_values)

                bins = bins % args.bins
                id2sketch[id].scatter_add_(1, bins, signs)

    elif args.method == "our-method":   

        for id in id2table:
            if id in skew_ids:
                id2sketch[id].construct(
                    node2batches, 
                    id_bigdata(id, node2batches, args.topk),
                    bin_hashes, 
                    sign_hashes,
                    query,
                    args)
            else:
                id2sketch[id].construct(
                    node2batches, 
                    {},
                    bin_hashes, 
                    sign_hashes,
                    query,
                    args)
        
    stream_time = stream_timer.stop()

    inference_timer = utils.Timer()

    print("Estimating cardinality...")
    if args.method == "count-conv":
        start_node = query.random_node()
        visited: Set[str] = set()

        estimates = combine_sketches(start_node, visited, query, id2sketch)
        estimates = torch.sum(estimates, dim=1)
    elif args.method == "our-method":
        start_node = query.random_node()
        visited: Set[str] = set()

        estimates = combine_splited_sketches(start_node, bin_hashes, sign_hashes, visited, query, id2sketch, args).get_sum()

    inference_time = inference_timer.stop()

    estimates = estimates.tolist()
    
    fieldnames = [
        "method",
        "query",
        "batch_size",
        "seed",
        "bins",
        "means",
        "medians",
        "estimate",
        "memory_usage",
        "init_time",
        "stream_time",
        "inference_time",
    ]

    # print("Estimates:", estimates[0],"\n")
    if args.method == "our-method":
        with open(f"./results/{args.query[:4]}{args.bins}", "a", newline="") as f:
            f.write(f"{estimates[0]}\n")
    else :
        with open(f"./results/{args.query[:4]}{args.bins}-conv", "a", newline="") as f:
            f.write(f"{estimates[0]}\n")

if __name__ == "__main__":
    print("start")
    experiment()
