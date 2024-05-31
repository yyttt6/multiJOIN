import numpy as np

x = np.random.rand((3))
y = np.random.rand((3))
z = np.random.rand((3))
x /= np.sum(x)
x *= 100
y /= np.sum(y)
y *= 100
z /= np.sum(z)
z *= 100
a = np.einsum('i,j,k->ijk',x,y,z)
a /= 100000
print(a)
print(np.sum(a))

b = np.random.rand(3, 3)
b /= np.sum(b)
b *= 9000
print(b)
p = np.sum(b,1)
q = np.sum(b,0)
print(p)
print(np.sum(p))
print(q)
print(np.sum(q))
# p = np.random.rand((3))
# q = np.random.rand((3))
# r = np.random.rand((3))
# p /= np.sum(p)
# p *= 2000
# q /= np.sum(q)
# q *= 2000
# r /= np.sum(r)
# r *= 2000
# m = np.random.rand((3))
# m /= np.sum(m)
# m *= 2000
# b = np.einsum('i,j,k,m->ijkm' ,p ,q ,r,m)
# b /= 8000000000
# print(b)
# print(np.sum(b))

#已经证明是等价的
c = np.einsum('kij,ij->k', a, b)
print(c)
print(np.sum(c))
m_ans = np.dot(y, p) * np.dot(z, q) / 9000000
print(m_ans)