import numpy as np
from io import StringIO

with open('influence.txt', 'r') as influence:
  holders = influence.readline().strip().split(' | ')
  inf_m = np.loadtxt(StringIO(influence.read().replace('_', '0')))

with open('interest.txt', 'r') as interest:
  assert(holders == interest.readline().strip().split(' | '))
  int_m = np.loadtxt(StringIO(interest.read().replace('_', '0')))

dim = len(holders)
ones = np.ones(dim)

mask = (np.array(
    [np.matmul(inf_m, ones), 
     np.matmul(int_m, ones)]
                ) >= np.array([[dim/2], [dim/2]])).prod(axis=0).astype(bool)

with open('result.txt', 'w') as result:
  for i in np.array(holders)[mask]:
    result.write(i)
    result.write('\n')
