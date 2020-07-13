

import numpy as np
import torch
from multiprocessing import Pool, Process

def read_txt(path):
    with open(path,'r') as f:
        f = f.readlines()
    ll = []
    for i in f:
        l = list(map(float, list(i.strip().split(' '))))
        l = np.array(l)
        ll.append(l)
    ll = np.array(ll)
    return torch.from_numpy(ll)

a = read_txt('95diff.txt')
b = read_txt('95rgb.txt')
c = read_txt('fanzhuan_tsm.txt')
d = read_txt('diff_85.txt')
e = read_txt('RGB_85.txt')
f = read_txt('diff_85_10_2_flip.txt')
g = read_txt('RGB_85_10_2_flip.txt')
h = read_txt('RGB_85_10_2.txt')
i = read_txt('diff_85_10_2.txt')
j = read_txt('diff_85_10.txt')
k = read_txt('RGB_85_10.txt')
l = read_txt('RGB_85_10___-1.txt')
m = read_txt('tsm_85_10.txt')
n = read_txt('diff_85_10___-1.txt')
o = read_txt('tsm_85_10___-1.txt')
p = read_txt('RGB_85_10_flip.txt')
q = read_txt('tsm_85_10_flip.txt')
r = read_txt('diff_85_10_concat.txt')
s = read_txt('tsm_85_10_concat.txt')
t = read_txt('RGB_85_10_concat.txt')
u = read_txt('RGB_85_10_multiscale.txt')
v = read_txt('diff_85_10_multiscale.txt')
w = read_txt('tsm_85_10_multiscale.txt')
x = read_txt('RGB_85_10_2_rand_ge2.txt')
y = read_txt('diff_85_10_2_rand_ge2.txt')
# z = read_txt('tsm_rgb.txt')
# zz = read_txt('tsm_rgb_::-1.txt')
# zzz = read_txt('tsm_rgb_flip.txt')
def result(torch_data,maxk = 5):
    _, pred = torch_data.data.topk(maxk, 1, True, True)

    pred = pred.cpu().numpy()+np.ones((1,))
    return pred.astype(np.int)

def my_test_acc(result):
    label = open('weitest.txt','r').readlines()
    num = 0
    for index, i in enumerate(result):
        if int(label[index].split(' ')[1])!=i[0]:
            num = num+1
    return (3783-num)/3783




def best(res,begin_num= [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y]):
    # print(res)
    the_best = []
    the_best_score =0
    for li in res:
        now = []
        r = []
        for i in li:
            now.append(begin_num[i])
            r.append(i)
        score = my_test_acc(result(sum(now)))
        if score> the_best_score:
            the_best_score = score
            the_best = r
    # print(the_best , the_best_score)
    return the_best , the_best_score

def find_best(num = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y]):
    begin_num = num
    num = list(range(len(num)))
    m = ppp(num)
    m.digui([],0)
    res = m.res
    # print(res)
    # res.pop()
    res = list(filter(lambda x:len(x)> 8 and len(x)<18,res))
    # print(res)
    the_best , the_best_score = best(res,begin_num)

if __name__=='__main__':
    print(1)
    processor = 50
    num = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y]
    # begin_num = num
    #

    num = [num[i] for i in [0, 1,2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
    for index,i in enumerate(num):
        a = my_test_acc(result(i))
        print(index,a)
