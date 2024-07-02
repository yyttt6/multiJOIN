import numpy as np
import mmh3
import heapq
import math
from copy import deepcopy

TOP = 0
SIZE = 2000

class Matrix:
    def __init__(self, lables: [], positions):
        self.format = 'Matrix'
        self.rows = 0
        self.size = SIZE
        self.lables = {}  # lables：特征对应的矩阵维度
        self.poss = {}  # poss： 特征在表中的位置
        self.dim = len(lables)  # dim：矩阵维度
        if self.dim > 2:
            raise Exception("dimension is too large")
        dims = [self.size for i in range(self.dim)]  # 维度向量，仅用于初始化numpy矩阵
        self.lables = {lables[i]: i for i in range(self.dim)}  # lables：特征对应的矩阵维度
        if not len(positions) == 0:
            self.poss = {positions[i]: lables[i] for i in range(self.dim)}  # poss： 特征在表中的位置
        self.data = np.zeros(dims)  # data: 矩阵列表，每一段为一个numpy矩阵

    def insert(self, infos, weight: np.int32 = 1):
        index = tuple(mmh3.hash(info, 101, signed=False) % self.size for info in infos)
        signs = [mmh3.hash(info, 102, signed=True) & 1 for info in infos]
        signs = [-1 if sign <= 0 else 1 for sign in signs]
        value = weight * np.int32(np.prod(signs))
        self.data[index] += value

    def generate(self, file):
        line = file.readline()
        while line:
            self.rows += 1
            infos = line.split('|')
            put_in = [infos[i] for i in range(len(infos)) if i in self.poss]
            self.insert(put_in)
            line = file.readline()

    def compress(self):
        if (self.dim == 2):
            ht = HalfTable(list(self.lables.keys()), {})
            ht.smldata = []
            ht.smldata.append(np.sum(self.data, 1))
            ht.smldata.append(np.sum(self.data, 0))
            ht.smldata = np.array(ht.smldata)
            ht.smldata.reshape(2, -1)
            ht.smldata[0] /= math.pow(np.sum(ht.smldata[0]), 1 / 2)
            ht.rows = self.rows
            return ht
        if (self.dim == 1):
            ht = HalfTable(list(self.lables.keys()), {})
            ht.smldata = self.data
            return ht
def M_Join(matrix1, matrix2):
    if matrix1.dim < matrix2.dim:
        matrix1, matrix2 = (matrix2, matrix1)
    temp_lables = matrix1.lables
    order = ''
    target = ''
    temp_dim = 0
    for x in temp_lables.keys():
        order += chr(temp_lables[x] + 97)
        target += chr(temp_lables[x] + 97)
    order += ','
    for x in matrix2.lables.keys():
        if x in temp_lables:
            order += chr(temp_lables[x] + 97)
            target = str.replace(target, chr(temp_lables[x] + 97), '')
            temp_lables.pop(x)
        else:
            order += chr(temp_dim + 65)
            target += chr(temp_dim + 65)
            temp_lables[x] = temp_dim
            temp_dim += 1
    order = order + '->' + target
    print(order)
    temp_dim += matrix1.dim
    new_matrix = Matrix(list(temp_lables.keys()), [])
    l1 = len(matrix1.data)
    l2 = len(matrix2.data)
    new_matrix.data = np.einsum(order, matrix1.data, matrix2.data)
    new_matrix.rows = 9
    return new_matrix


class HalfTable:
    def __init__(self, lables: [], positions):
        self.format = 'HalfTable'
        self.rows = 0;
        self.size = SIZE
        self.dim = len(lables)  # dim：数据维度
        self.lables = {lables[i]: i for i in range(self.dim)}  # lables：特征对应的矩阵维度
        if not len(positions) == 0:
            self.poss = {positions[i]: lables[i] for i in range(self.dim)}  # poss： 特征在表中的位置
        else:
            self.poss = {}
        self.bigdata = {}
        self.smldata = np.zeros((self.dim, self.size))

    def generate(self, file):
        line = file.readline()
        temp = {}
        while line:
            self.rows += 1
            infos = line.split('|')
            put_in = tuple([infos[i] for i in range(len(infos)) if i in self.poss])
            if put_in not in temp:
                temp[put_in] = 1
            else:
                temp[put_in] += 1
            line = file.readline()
        hplist = [(-freq, key) for key, freq in temp.items()]
        heapq.heapify(hplist)
        for i in range(TOP):
            (freq, tup) = hplist[i]
            self.bigdata[tup] = -freq
        for i in range(self.dim):
            for j in range(TOP, len(hplist), 1):
                (freq, tup) = hplist[j]
                index = mmh3.hash(tup[i], 101, signed=False) % self.size
                sign = mmh3.hash(tup[i], 102, signed=True) & 1
                if sign == 0:
                    sign = -1
                self.smldata[i, index] += sign * (-freq)


def H_Join(halftable1, halftable2):
    if (halftable1.dim < halftable2.dim):
        halftable1, halftable2 = (halftable2, halftable1)
    temp_cardinality = 1  # 联通基数
    inter_lables = []  # 交集标签
    left_lables1 = {}  # 左表标签
    left_lables2 = {}  # 右表标签
    for x in halftable2.lables.keys():
        if x in halftable1.lables.keys():
            inter_lables.append(x)
        else:
            left_lables2[x] = halftable2.lables[x]
    for x in halftable1.lables.keys():
        if not x in inter_lables:
            left_lables1[x] = halftable1.lables[x]
    # 获得三个集合

    # 计算联通基数
    for x in inter_lables:
        temp_cardinality *= np.dot((halftable1.smldata[halftable1.lables[x]]), (halftable2.smldata[halftable2.lables[x]]))
    temp_cardinality /= np.power(halftable1.rows, (len(inter_lables)-1))
    temp_cardinality /= np.power(halftable2.rows, (len(inter_lables)-1))
    # 已经确定，这种操作等价于矩阵相乘

    temp_lables = deepcopy(left_lables1)
    temp_lables.update(left_lables2)  # 新表的标签就是剩余标签之和
    assert len(temp_lables) < halftable1.dim + halftable2.dim
    i = 0
    for key in temp_lables.keys():
        temp_lables[key] = i
        i += 1  # 至此，获得新表的标签
    temp_smldata = []
    temp_rows = temp_cardinality
    for key in temp_lables.keys():  # 计算新表的sml_data
        if key in halftable1.lables:
            temp_smldata.append(np.copy(halftable1.smldata[halftable1.lables[key]]))
        if key in halftable2.lables:
            temp_smldata.append(np.copy(halftable2.smldata[halftable2.lables[key]]))
    if (temp_smldata == []):
        temp_smldata = 1
    new_ht = HalfTable(list(temp_lables.keys()), {})
    new_ht.smldata = np.array(temp_smldata) * temp_cardinality
    new_ht.rows = temp_rows
    # new_ht.bigdata = {}  # 计算新表的big_data
    # for big1 in halftable1.bigdata.keys():
    #     for big2 in halftable2.bigdata.keys():
    #         flag = True
    #         for x in inter_lables:
    #             if (not big1[halftable1.lables[x]] == big2[halftable2.lables[x]]):
    #                 flag = False
    #                 break
    #         if flag == True:
    #             key = []
    #             for x in inter_lables:
    #                 key.append(big1[halftable1.lables[x]])
    #                 key.append(big2[halftable2.lables[x]])
    #             key = tuple(key)
    #             if new_ht.bigdata.get(key) == None:
    #                 new_ht.bigdata[key] = halftable1.bigdata[big1] * halftable2.bigdata[big2]
    #             else:
    #                 new_ht.bigdata[key] += halftable1.bigdata[big1] * halftable2.bigdata[big2]
    return new_ht


def Join(table1, table2):
    if (table1.format == table2.format == 'Matrix'):
        return M_Join(table1, table2)
    if (table1.format == 'Matrix'):
        return H_Join(table1.compress(), table2)
    if (table2.format == 'Matrix'):
        return H_Join(table1, table2.compress())
    if table1.dim < table2.dim:
        return H_Join(table1, table2)


def init(lebal, poss):
    if (len(lebal) <= 2):
        return Matrix(lebal, poss)
    else:
        return HalfTable(lebal, poss)


llebal = ['ORDERKEY', 'PARTKEY', 'SUPPKEY']
lposs = [0, 1, 2]
plebal = ['PARTKEY', 'SUPPKEY']
pposs = [0, 1]
olebal = ['ORDERKEY']
oposs = [0]
lfile = open('./lineitem.tbl', 'r')
pfile = open('./partsupp.tbl', 'r')
ofile = open('./order.tbl', 'r')

lht = init(llebal, lposs)
pht = init(plebal, pposs)
oht = init(olebal, oposs)
lht.generate(lfile)
pht.generate(pfile)
oht.generate(ofile)

print("HalfTable Start Join !")
ans_ht = Join(Join(pht, lht), oht)
print("HalfTable answer:" + str(ans_ht.smldata))