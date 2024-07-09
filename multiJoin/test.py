import numpy as np

a = np.array([0,0,0])
print([-1 if i <= 0 else 1 for i in a])

# b = np.random.rand(3, 3, 3)
# b /= np.sum(b)
# b *= 81
# print(b)
# r = np.sum(np.sum(b,2),1)#0
# p = np.sum(np.sum(b,2),0)#1
# q = np.sum(np.sum(b,0),0)#2
# print(r)
# print(np.sum(r))
# print(p)
# print(np.sum(p))
# print(q)
# print(np.sum(q))
#
# a = np.random.rand(3,3)
# a /= np.sum(a)
# a *=27
# x = np.sum(a,0)
# y = np.sum(a,1)
# print(x)
# print(np.sum(x))
# print(y)
# print(np.sum(y))
#
# #已经证明是等价的
# c = np.einsum('kli,ij->klj', b, a)
# print(c)
# print(np.sum(c))
# m_ans =0
# print(m_ans)