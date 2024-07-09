import numpy as np
import mmh3
import heapq
import math
from copy import deepcopy

class CorrectTable:
    def __init__(self, lables: []):
        self.dim = len(lables)
        self.lables = lables
        self.rows = 0
        self.data = {}

    def generate(self, filename, positions: [], weight: np.int32 = 1):
        file = open(filename, 'r')
        self.data = {}
        line = file.readline()
        self.rows += 1
        while line:
            infos = line.split('|')
            put_in = tuple([infos[i] for i in positions])
            if put_in not in self.data:
                self.data[put_in] = 1
            else:
                self.data[put_in] += 1
            line = file.readline()
            self.rows += 1
        file.close()


def join(t1, t2):
    #求t1和t2的交集
    temp_cardinality = 1  # 联通基数
    inter_lables1 = {}  # 交集标签
    inter_lables2 = {}  # 交集标签
    left_lables1 = {}  # 左表标签
    left_lables2 = {}  # 右表标签
    for x in t1.lables:
        if x in t2.lables:
            inter_lables1[x] = t1.lables.index(x)
            inter_lables2[x] = t2.lables.index(x)
        else:
            left_lables1[x] = t1.lables.index(x)
        for x in t2.lables:
            if x not in t1.lables: left_lables2[x] = t2.lables.index(x)
    new_table = CorrectTable(list(left_lables1.keys())+list(left_lables2.keys()))
    t2_handles = {}
    for tup, freq in t2.data.items():
        temp_handle = tuple([tup[i] for i in inter_lables2.values()])
        if temp_handle not in t2_handles:
            t2_handles[temp_handle] = [tuple([tuple([tup[i] for i in left_lables2.values()]),freq])]
        else:
            t2_handles[temp_handle].append(tuple([tuple([tup[i] for i in left_lables2.values()]),freq]))
    for tup1, freq1 in t1.data.items():
        temp_handle = tuple([tup1[i] for i in inter_lables1.values()])
        temp_new_handle = [tup1[i] for i in left_lables1.values()]
        if temp_handle in t2_handles:
            for tup2, freq2 in t2_handles[temp_handle]:
                new_tup = tuple(temp_new_handle + list(tup2))
                if new_tup not in new_table.data:
                    new_table.data[new_tup] = freq1 * freq2
                else:
                    new_table.data[new_tup] += freq1 * freq2
    return new_table


customer_f = './customer.tbl'
lineitem_f = './lineitem.tbl'
nation_f = './nation.tbl'
orders_f = './order.tbl'
part_f = './part.tbl'
partsupp_f = './partsupp.tbl'
region_f = './region.tbl'
supplier_f = './supplier.tbl'

# part_l = ['PARTKEY']
# part_p = [0]
# part_t = CorrectTable(part_l)
# part_t.generate(part_f, part_p)

partsupp_l = ['PARTKEY', 'SUPPKEY']
partsupp_p = [0, 1]
partsupp_t = CorrectTable(partsupp_l)
partsupp_t.generate(partsupp_f, partsupp_p)

lineitem_l = ['ORDERKEY','PARTKEY','SUPPKEY']
lineitem_p = [0,1,2]
lineitem_t = CorrectTable(lineitem_l)
lineitem_t.generate(lineitem_f, lineitem_p)

# orders_l = ['ORDERKEY', 'CUSTKEY']
# orders_p = [0, 1]
# orders_t = CorrectTable(orders_l)
# orders_t.generate(orders_f, orders_p)

orders_l = ['ORDERKEY']
orders_p = [0]
orders_t = CorrectTable(orders_l)
orders_t.generate(orders_f, orders_p)

# customer_l = ['CUSTKEY', 'NATIONKEY']
# customer_p = [0, 3]
# customer_t = CorrectTable(customer_l)
# customer_t.generate(customer_f, customer_p)

# supplier_l = ['NATIONKEY']
# supplier_p = [3]
# supplier_t = CorrectTable(supplier_l)
# supplier_t.generate(supplier_f, supplier_p)

join_queue = [#part_t,
              partsupp_t, lineitem_t, orders_t
              #, customer_t,supplier_t
              ]

print("CorrectTable Start Join !")
ans = join_queue[0]
for i in range(1, len(join_queue)):
    ans = join(join_queue[i], ans)
print("CorrectTable answer:" + str(ans.data))
















