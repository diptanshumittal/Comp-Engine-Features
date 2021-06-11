n = int(input())
for qwer in range(n):
    p = int(input())
    li = [0]*200001
    temp = list(map(int,input().split(" ")))
    max = 1
    dis = 0
    for i in temp:
        if(li[i]==0):
            li[i] = 1
            dis+=1
            #print("here")
        else:
            li[i] +=1
            if(max<li[i]):
                max = li[i]
    #print("max",max ,  " " , dis)
    if(p==1):
        print(0)
    elif(dis<max):
        print(dis)
    elif(max==dis):
        print(max-1)
    else:
        print(max)
    #print("ans ")