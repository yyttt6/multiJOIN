import numpy as np
import mmh3
import heapq
import math
from copy import deepcopy

SIZE = 2000
TOP = 10


class Table:
    def __init__(self, lables: []):
        self.dim = len(lables)
        self.lables = lables
        self.rows = 0
        if self.dim == 0:
            self.data = 0
        elif self.dim == 2:
            self.data = np.zeros([SIZE, SIZE])
        else:
            self.data = np.zeros([self.dim, SIZE])
        self.bigdata = {}

    def generate(self, filename, positions: [], weight: np.int32 = 1):
        file = open(filename, 'r')
        if self.dim == 1:
            line = file.readline()
            self.rows += 1
            while line:
                infos = line.split('|')
                index = mmh3.hash(infos[positions[0]], 101, signed=False) % SIZE
                sign = mmh3.hash(infos[positions[0]], 102, signed=True) & 1
                if sign == 0: sign = -1
                self.data[0][index] += weight * np.int32(sign)
                line = file.readline()
                self.rows += 1
        elif self.dim == 2:
            line = file.readline()
            self.rows += 1
            while line:
                infos = line.split('|')
                index = tuple([mmh3.hash(infos[i], 101, signed=False) % SIZE for i in positions])
                signs = [mmh3.hash(infos[i], 102, signed=True) & 1 for i in positions]
                sign = np.prod([-1 if i <= 0 else 1 for i in signs])
                self.data[index] += weight * np.int32(sign)
                line = file.readline()
                self.rows += 1
        else:
            temp = {}
            line = file.readline()
            self.rows += 1
            while line:
                infos = line.split('|')
                put_in = tuple([infos[i] for i in positions])
                if put_in not in temp:
                    temp[put_in] = 1
                else:
                    temp[put_in] += 1
                line = file.readline()
                self.rows += 1
            hplist = [(-freq, key) for key, freq in temp.items()]
            heapq.heapify(hplist)
            for i in range(TOP):
                (freq, tup) = hplist[i]
                self.bigdata[tup] = -freq
                self.rows += freq
            for j in range(TOP, len(hplist), 1):
                (freq, tup) = hplist[j]
                for i in range(self.dim):
                    index = mmh3.hash(tup[i], 101, signed=False) % SIZE
                    sign = mmh3.hash(tup[i], 102, signed=True) & 1
                    if sign == 0: sign = -1
                    self.data[i, index] += sign * (-freq)
        file.close()


def join(t1, t2):
    if t1.dim > t2.dim:  t1, t2 = (t2, t1)
    # 分四种情况 1*2 2*2 2*n m*n
    # 1*2
    if t1.dim == 1 and t2.dim == 2:
        if not t1.bigdata == {}:
            for tup, freq in t1.bigdata.items():
                index = mmh3.hash(tup[0], 101, signed=False) % SIZE
                sign = mmh3.hash(tup[0], 102, signed=True) & 1
                if sign == 0: sign = -1
                t1.data[index] += sign * freq
        if not t2.bigdata == {}:
            for tup, freq in t2.bigdata.items():
                index = [mmh3.hash(tup[0], 101, signed=False) % SIZE, mmh3.hash(tup[1], 101, signed=False) % SIZE]
                signs = [mmh3.hash(tup[0], 102, signed=True) & 1, mmh3.hash(tup[1], 102, signed=True) & 1]
                sign = np.prod([-1 if i <= 0 else 1 for i in signs])
                t2.data[tuple(index)] += sign * freq
        t1.data = t1.data.reshape((SIZE,))
        same_id = t2.lables.index(t1.lables[0])
        new_lable = deepcopy(t2.lables)
        new_lable.pop(same_id)
        new_table = Table(new_lable)
        if same_id == 0:
            new_table.data = np.einsum('i,ij->j', t1.data, t2.data)
        else:
            new_table.data = np.einsum('i,ji->j', t1.data, t2.data)
    # 2*2 join1或join2
    elif t1.dim == 2 and t2.dim == 2:
        if not t1.bigdata == {}:
            for tup, freq in t1.bigdata.items():
                index = [mmh3.hash(tup[0], 101, signed=False) % SIZE, mmh3.hash(tup[1], 101, signed=False) % SIZE]
                signs = [mmh3.hash(tup[0], 102, signed=True) & 1, mmh3.hash(tup[1], 102, signed=True) & 1]
                sign = np.prod([-1 if i <= 0 else 1 for i in signs])
                t1.data[tuple(index)] += sign * freq
        if not t2.bigdata == {}:
            for tup, freq in t2.bigdata.items():
                index = [mmh3.hash(tup[0], 101, signed=False), mmh3.hash(tup[1], 101, signed=False)]
                signs = [mmh3.hash(tup[0], 102, signed=True) & 1, mmh3.hash(tup[1], 102, signed=True) & 1]
                sign = np.prod([-1 if i <= 0 else 1 for i in signs])
                t2.data[tuple(index)] += sign * freq
        # join2
        if t1.lables[0] == t2.lables[0] and t1.lables[1] == t2.lables[1]:
            new_table = Table([])
            new_table.data = np.einsum('ij,ij->', t1.data, t2.data)
        elif t1.lables[0] == t2.lables[1] and t1.lables[1] == t2.lables[0]:
            new_table = Table([])
            new_table.data = np.einsum('ij,ji->', t1.data, t2.data)
        # join1
        elif t1.lables[0] in t2.lables:
            id1 = t2.lables.index(t1.lables[0])
            if id1 == 0:
                new_table = Table([t1.lables[1], t2.lables[1]])
                new_table.data = np.einsum('ij,ik->jk', t1.data, t2.data)
            else:
                new_table = Table([t1.lables[1], t2.lables[0]])
                new_table.data = np.einsum('ij,ki->jk', t1.data, t2.data)
        elif t1.lables[1] in t2.lables:
            id1 = t2.lables.index(t1.lables[1])
            if id1 == 0:
                new_table = Table([t1.lables[0], t2.lables[1]])
                new_table.data = np.einsum('ij,jk->ik', t1.data, t2.data)
            else:
                new_table = Table([t1.lables[0], t2.lables[0]])
                new_table.data = np.einsum('ij,kj->ik', t1.data, t2.data)
    # 2*n 分成join2和join1
    elif t1.dim == 2:
        if not t1.bigdata == {}:
            for tup, freq in t1.bigdata.items():
                index = [mmh3.hash(tup[0], 101, signed=False) % SIZE, mmh3.hash(tup[1], 101, signed=False) % SIZE]
                signs = [mmh3.hash(tup[0], 102, signed=True) & 1, mmh3.hash(tup[1], 102, signed=True) & 1]
                sign = np.prod([-1 if i <= 0 else 1 for i in signs])
                t1.data[tuple(index)] += sign * freq
        # join2 分为产生2和产生非2
        if t1.lables[0] in t2.lables and t1.lables[1] in t2.lables:
            id1 = t2.lables.index(t1.lables[0])
            id2 = t2.lables.index(t1.lables[1])
            temp_lables = deepcopy(t2.lables)
            temp_lables.pop(id1)
            new_id2 = temp_lables.index(t1.lables[1])  # new_id2是移除id1之后的id2
            temp_lables.pop(new_id2)
            new_table = Table(temp_lables)
            temp_martix = np.einsum('i,j->ij', t2.data[id1], t2.data[id2])
            temp_cardinality = np.einsum('ij,ij->', t1.data, temp_martix) / t2.rows
            new_table.data = temp_cardinality * np.array([t2.data[t2.lables.index(i)] for i in temp_lables])
            new_table.rows = temp_cardinality
            # 产生2
            if new_table.dim == 2:
                new_table.data = np.einsum('i,j->ij', new_table.data[0], new_table.data[1]) / new_table.rows
                for tup, freq in t2.bigdata.items():
                    index = [mmh3.hash(tup[id1], 101, signed=False) % SIZE,
                             mmh3.hash(tup[id2], 101, signed=False) % SIZE]
                    signs = [mmh3.hash(tup[id1], 102, signed=True) & 1, mmh3.hash(tup[id2], 102, signed=True) & 1]
                    sign = np.prod([-1 if i <= 0 else 1 for i in signs])
                    new_table.data[tuple(index)] += sign * freq * t1.data[tuple(index)]
                    new_table.rows += sign * freq * t1.data[tuple(index)]
            # 产生非2
            else:
                tbigdata = {}
                for tup, freq in t2.bigdata.items():
                    index = [mmh3.hash(tup[id1], 101, signed=False) % SIZE,
                             mmh3.hash(tup[id2], 101, signed=False) % SIZE]
                    signs = [mmh3.hash(tup[id1], 102, signed=True) & 1, mmh3.hash(tup[id2], 102, signed=True) & 1]
                    sign = np.prod([-1 if i <= 0 else 1 for i in signs])
                    tup = list(tup)
                    tup.pop(id1)
                    tup.pop(new_id2)
                    tup = tuple(tup)
                    freq = freq * sign * t1.data[tuple(index)]
                    if freq > 0:
                        if tup not in tbigdata:
                            tbigdata[tup] = freq
                        else:
                            tbigdata[tup] += freq
        # join1 只可能产生非2 由于2没有大流，所以新table的大流的tuple写不出来，只能放弃
        else:
            for tup, freq in t2.bigdata.items():
                for i in range(t2.dim):
                    x = tup[i]
                    index = mmh3.hash(x, 101, signed=False) % SIZE
                    sign = mmh3.hash(x, 102, signed=True) & 1
                    if sign == 0: sign = -1
                    t2.data[i][index] += sign * freq
            id1 = 0
            if t1.lables[1] in t2.lables: id1 = 1
            id2 = t2.lables.index(t1.lables[id1])
            temp_lables = deepcopy(t2.lables)
            temp_lables.pop(id2)
            temp_lables.insert(0, t1.lables[1 - id1])
            new_table = Table(temp_lables)
            new_table.data = [t1.data[1 - id1]]
            for i in range(1, len(temp_lables)):
                new_table.data.append(t2.data.index(temp_lables[i]))
            x = np.sum(t1.data, axis=1)
            y = np.sum(t1.data, axis=0)
            if id1 == 0:
                temp_cardinality = np.dot(x, t2.data[id2])
            else:
                temp_cardinality = np.dot(y, t2.data[id2])
            new_table.data = temp_cardinality * np.array(new_table.data)
            new_table.rows = temp_cardinality
    # m*n 分为生成2和不生成2
    else:
        if t1.dim == 1: t1.data = np.reshape(t1.data,(SIZE,))
        if t2.dim == 1: t2.data = np.reshape(t2.data, (SIZE,))
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
        # 获得三个集合
        for x in inter_lables1.keys():
            temp_cardinality *= np.dot((t1.data[inter_lables1[x]]),
                                       (t2.data[inter_lables2[x]]))
        temp_cardinality /= np.power(t1.rows, (len(inter_lables1) - 1))
        temp_cardinality /= np.power(t2.rows, (len(inter_lables1) - 1))
        # 已经确定，这种操作等价于矩阵相乘
        new_table = Table(list(left_lables1.keys()) + list(left_lables2.keys()))
        new_table.rows = temp_cardinality
        new_table.data = np.array([t1.data[id] for x, id in left_lables1.items()] + [t2.data[id] for x, id in
                                                                                     left_lables2.items()]) * temp_cardinality
        t1_handles = []
        t2_handles = []
        t1_lefts = []
        t2_lefts = []
        for tup, freq in t1.bigdata.items():
            t1_handles.append([tup[i] for i in inter_lables1.values()])
            t1_lefts.append([tup[i] for i in left_lables1.values()])
        for tup, freq in t2.bigdata.items():
            t2_handles.append([tup[i] for i in inter_lables2.values()])
            t2_lefts.append([tup[i] for i in left_lables2.values()])
        for i in range(len(t1_handles)):
            for j in range(len(t2_handles)):
                if t1_handles[i] == t2_handles[j]:
                    tup = tuple(t1_lefts[i] + t2_lefts[j])
                    freq = t1.bigdata.values()[i] * t2.bigdata.values()[j]
                    if tup not in new_table.bigdata:
                        new_table.bigdata[tup] = freq
                    else:
                        new_table.bigdata[tup] += freq

        if new_table.dim == 0: new_table.data = new_table.rows
        if new_table.dim == 2:
            new_table.data = np.einsum('i,j->ij', new_table.data[0], new_table.data[1]) / new_table.rows
            for tup, freq in new_table.bigdata.items():
                index = [mmh3.hash(tup[0], 101, signed=False) % SIZE, mmh3.hash(tup[1], 101, signed=False) % SIZE]
                signs = [mmh3.hash(tup[0], 102, signed=True) & 1, mmh3.hash(tup[1], 102, signed=True) & 1]
                sign = np.prod([-1 if i <= 0 else 1 for i in signs])
                new_table.data[tuple(index)] += sign * freq
                new_table.rows += sign * freq
    return new_table


customer_f = './customer.tbl'
lineitem_f = './lineitem.tbl'
nation_f = './nation.tbl'
orders_f = './order.tbl'
part_f = './part.tbl'
partsupp_f = './partsupp.tbl'
region_f = './region.tbl'
supplier_f = './supplier.tbl'

part_l = ['PARTKEY']
part_p = [0]
partsupp_l = ['PARTKEY', 'SUPPKEY']
partsupp_p = [0, 1]
lineitem_l = ['ORDERKEY', 'SUPPKEY']
lineitem_p = [0, 2]
orders_l = ['ORDERKEY', 'CUSTKEY']
orders_p = [0, 1]
customer_l = ['CUSTKEY', 'NATIONKEY']
customer_p = [0, 3]
supplier_l = ['NATIONKEY']
supplier_p = [3]

part_t = Table(part_l)
part_t.generate(part_f, part_p)
partsupp_t = Table(partsupp_l)
partsupp_t.generate(partsupp_f, partsupp_p)
lineitem_t = Table(lineitem_l)
lineitem_t.generate(lineitem_f, lineitem_p)
orders_t = Table(orders_l)
orders_t.generate(orders_f, orders_p)
customer_t = Table(customer_l)
customer_t.generate(customer_f, customer_p)
supplier_t = Table(supplier_l)
supplier_t.generate(supplier_f, supplier_p)
join_queue = [part_t, partsupp_t, lineitem_t, orders_t, join(customer_t, supplier_t)]

print("Table Start Join !")
ans = join_queue[0]
for i in range(1, len(join_queue)):
    ans = join(join_queue[i], ans)
print("Table answer:" + str(int(ans.data)))
# print("Table Start Join !")
# ans_t = join(join(pt,lt),ot)
# print("Table answer:" + str(int(ans_t.data)))
















