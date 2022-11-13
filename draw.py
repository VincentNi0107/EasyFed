import matplotlib.pyplot as plt
import pickle
with open('lowrate-1.0comfeq-1epo10', 'rb') as f:
    fedavg10 = pickle.load(f)
# with open('lowrate-1.0comfeq-2epo10', 'rb') as f:
#     fedavg20 = pickle.load(f)
# with open('lowrate-1.0comfeq-4epo10', 'rb') as f:
#     fedavg40 = pickle.load(f)
# with open('lowrate-2.0comfeq-2epo10', 'rb') as f:
#     fedcc20_2 = pickle.load(f)
# # with open('lowrate-0.0comfeq-2epo5co', 'rb') as f:
# #     fedhcco = pickle.load(f)
with open('lowrate-0.3comfeq-2', 'rb') as f:
    fedhc032 = pickle.load(f)
with open('lowrate-0.5comfeq-2', 'rb') as f:
    fedhc052 = pickle.load(f)
with open('lowrate-0.8comfeq-2', 'rb') as f:
    fedhc082 = pickle.load(f)
with open('lowrate-0.3comfeq-4', 'rb') as f:
    fedhc034 = pickle.load(f)    
with open('lowrate-0.5comfeq-4', 'rb') as f:
    fedhc054 = pickle.load(f)   
with open('lowrate-0.8comfeq-4', 'rb') as f:
    fedhc084 = pickle.load(f)   
# with open('lowrate-1.0comfeq-4', 'rb') as f:
#     fedhc104 = pickle.load(f) 
with open('lowrate-0.8comfeq-6', 'rb') as f:
    fedhc086 = pickle.load(f) 
with open('lowrate-0.8comfeq-8', 'rb') as f:
    fedhc088 = pickle.load(f) 
    
x=range(len(fedavg10))
plt.plot(x, fedavg10,label='fedavg/loc_epo=10')
# plt.plot(x, fedavg20,label='fedavg/loc_epo=20')
# plt.plot(x, fedavg40,label='fedavg/loc_epo=40')
# plt.plot(x, fedcc20_2,label='fedcc/loc_epo=20/cls_com_epo=10')
# plt.plot(x, fedhcco[:60],label='co/2/5')
plt.plot(x, fedhc032,label='sha_per=0.3/every 2 round')
plt.plot(x, fedhc052,label='sha_per=0.5/every 2 round')
plt.plot(x, fedhc082,label='sha_per=0.8/every 2 round')
plt.plot(x, fedhc034,label='sha_per=0.3/every 4 round')
plt.plot(x, fedhc054,label='sha_per=0.5/every 4 round')
plt.plot(x, fedhc084,label='sha_per=0.8/every 4 round')
# plt.plot(x, fedhc104,label='1.0/4')
plt.plot(x, fedhc086,label='sha_per=0.8/every 6 round')
plt.plot(x, fedhc088,label='sha_per=0.8/every 8 round')
# print(fedavg)
# print(fedhcco)
plt.xlabel('comm_round')
plt.ylabel('test_acc')
plt.legend()
plt.savefig("a.png")
print(max(fedavg10))
print(max(fedhc032))
print(max(fedhc052))
print(max(fedhc082))
print(max(fedhc034))
print(max(fedhc054))
print(max(fedhc084))
print(max(fedhc086))
print(max(fedhc088))