import numpy as np
data_iter = None
print("Loading", )
with open('test', 'r') as f:
    data_iter = f.readlines()

for idx, line in enumerate(data_iter):
    print(idx, line)
    line = [ln.split(",") for ln in line.split()]
    line = np.array(line)
    # print("line: ",line,"\n\n")
    logkey = line.squeeze()
    print("logkey: ",logkey)
    tim = np.zeros(logkey.shape)
    logkeys, times = [logkey.tolist()], [tim.tolist()]
    print("logkeys: ",logkeys)
    print("times: ",times)