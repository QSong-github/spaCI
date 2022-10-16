import os
import sys

f = open('log/parameters.txt', 'r')
lines = f.readlines()

bestf1 = 0
dicts = {}
for line in lines:
	conts = line.strip().split(',')
	k = int(conts[0].split(':')[1][1:])
	p = float(conts[1].split(':')[1][1:])
	alpha = float(conts[2].split(':')[1][1:])

	f1 = float(conts[3].split(':')[1][1:])
	dicts[f1] = (k, p, alpha)
	if bestf1 < f1:
		bestf1 = f1

print('find parameters set is: ', dicts[bestf1])

