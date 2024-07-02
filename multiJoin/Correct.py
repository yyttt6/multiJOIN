import numpy as np

class CorrectTable:
    def __init__(self, lables:[], positions):
        self.lables = {}
        self.poss = {}
        self.dim = len(lables)
        for i in range(self.dim):
            self.lables[lables[i]] = i
        if not len(positions) == 0:
            for i in range(self.dim):
                self.poss[positions[i]] = lables[i]
        self.data = {}
    
    def generate(self, file):
        line = file.readline()
        while line:
            infos = line.split('|')
            put_in = [infos[i] for i in range(self.dim) if i in self.poss]
            if self.data.get(tuple(put_in)) == None:
                self.data[tuple(put_in)] = 1
            else:
                self.data[tuple(put_in)] += 1
            line = file.readline()
            
def CorrectJoin(ctable1: CorrectTable, ctable2: CorrectTable):
    # Step 1: Determine the common and unique labels between the two tables
    shared_labels = set(ctable1.lables).intersection(set(ctable2.lables))
    unique_labels_c1 = set(ctable1.lables).difference(shared_labels)
    unique_labels_c2 = set(ctable2.lables).difference(shared_labels)
    # Step 2: Create a new CorrectTable with combined labels
    combined_labels = list(unique_labels_c1) + list(unique_labels_c2)
    combined_labels.sort()
    new_correct_table = CorrectTable(combined_labels, [])
    # Step 3: Merge the data from both tables into the new table
    for key1, value1 in ctable1.data.items():
        for key2, value2 in ctable2.data.items():
            # Check if keys share the same values for the shared labels
            flag = 1
            for slable in shared_labels:
                id1 = ctable1.lables[slable]
                id2 = ctable2.lables[slable]
                ch1 = key1[id1]
                ch2 = key2[id2]
                if not ch1 == ch2:
                    flag = 0
                    break
            if flag == 1:
            #if all(key1[ctable1.lables[slable]] == key2[ctable2.lables[slable]] for slable in shared_labels):
                merged_key = [key1[ctable1.lables[slable]] for slable in unique_labels_c1] + [key2[ctable2.lables[slable]] for slable in unique_labels_c2]
                merged_key.sort()
                new_value = value1 * value2
                if new_correct_table.data.get(tuple(merged_key)) == None:new_correct_table.data[tuple(merged_key)] = new_value
                else:new_correct_table.data[tuple(merged_key)] += new_value
    return new_correct_table

llebal = ['ORDERKEY','PARTKEY','SUPPKEY']
lposs = [0,1,2]
plebal = ['PARTKEY','SUPPKEY']
pposs = [0,1]
olebal = ['ORDERKEY']
oposs = [0]
lfile = open('./lineitem.tbl', 'r')
pfile = open('./partsupp.tbl', 'r')
ofile = open('./order.tbl', 'r')

ltable = CorrectTable(llebal,lposs)
ptable = CorrectTable(plebal,pposs)
otable = CorrectTable(olebal,oposs)
ltable.generate(lfile)
ptable.generate(pfile)
otable.generate(ofile)

print("Table start join !")
ans_c = CorrectJoin(CorrectJoin(ptable, ltable),otable)
print("Correct Table answer:" + str(ans_c.data.get(())))