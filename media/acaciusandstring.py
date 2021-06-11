import math
from collections import Counter 
temp = int(input())
for eqw in range(temp):
    t = []
    for rew in range(2):
        y = int(input())
        li = list(map(int, input().split(" ")))
        for i in range(1,y):
            li[i] += li[i-1]
        t.append(max(li))
    print(max( max(0,t[0]) , max(t[1],t[0]+t[1])))