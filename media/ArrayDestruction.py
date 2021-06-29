def rec(arr,x):
    x = x-arr[-1]
    arr = arr[:-1]
    if( x in arr):
        arr.remove(x)
        return True, arr
    else:
        return False, arr
for rewr in range(int(input())):
    n = int(input())
    b = list(map(int,input().split(" ")))
    b.sort()
    ye = 0
    for fir in range(len(b)-1):
        flag = 0
        temp1 = b.copy()
        ans = []
        ans.append([temp1[-1],temp1[fir]])
        pre = temp1[-1]
        temp1.pop(fir)
        temp1 = temp1[:-1]
        while len(temp1)>0:
            tr = temp1[-1]
            #print(temp1)
            cr, temp1 = rec(temp1,pre)
            if(cr):
                ans.append([tr,pre-tr])
                pre = tr
            else:
                flag = 1
                break
        if flag==0:
            ye = 1
            print("YES")
            print(ans[0][0]+ans[0][1])
            for i in ans:
                print(i[0],i[1])
            break
    if(ye==0):
        print("NO")