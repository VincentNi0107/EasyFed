def findrefclient(lastPart,partList,round,clientIdx):
    commonClient=[]
    lastRound=lastPart[clientIdx]
    if lastRound==-1:
        return commonClient
    lastRoundPart=partList[lastRound]
    RoundPart=partList[round]
    for client in lastRoundPart:
        if client in RoundPart and client!=clientIdx:
            commonClient.append(client)
    return commonClient 

last=[-1,0,0]
partList=[]
partList.append([1,2])
partList.append([0,1])
cl=findrefclient(last,partList,1,2)
print(cl)