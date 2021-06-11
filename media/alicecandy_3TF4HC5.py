rew = int(input())
for i in range(rew):
    n = int(input())
    temp = list(map(int,input().split(" ")))

    cum = []
    cum.append(temp[0])
    for i in range(1,n):
        cum.append(cum[i-1]+temp[i])
    total = cum[n-1]
    rcum = []
    rcum.append(total)
    for i in range(1,n):
        rcum.append(rcum[i-1]-temp[i-1])
    rcum.append(0)
    op = -1
    op1 = n
    print(cum)
    print(rcum)

    pre = 0 
    al = 0
    bb = 0
    i=0
    while len(temp)>0 and op+1<op1:
        print(op , "  trtre " , op1)
        if(i%2==0):
            low = op+1
            high = op1-1
            find = al+pre
            if(cum[high]<find):
                al = cum[high]
                i+=1
            while True:
                print(low,"  ",high , "   1")
                mid = int((low+high)/2)
                if(low+1==high):
                    break
                elif(cum[mid]<=find):
                    low = mid
                else:
                    high = mid
            al = cum[high]
            pre = cum[high]-cum[op]
            op = high
            i+=1
        if(i%2==1):
            low = op+1
            high = op1-1
            find = bb+pre
            if(rcum[low]<find):
                al = rcum[low]
                i+=1
            while True:
                print(low , "  " , high)
                mid = int((low+high)/2)
                if(low+1>=high):
                    break
                elif(cum[mid]<=find):
                    high = mid
                else:
                    low = mid
            bb = rcum[low]
            pre = rcum[low]-rcum[op1]
            op1 = low
            i+=1
    print(i,end=" ")
    print(al,end=" ")
    print(bb)