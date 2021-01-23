import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

data = [pd.read_csv('Evals/sql_server.csv', sep=","),
        pd.read_csv('Evals/design_patterns.csv', sep=","),
        pd.read_csv('Evals/unit_testing.csv', sep=","),
        pd.read_csv('Evals/visual_studio.csv', sep=","),
        pd.read_csv('Evals/xamarin.csv', sep=","),
        pd.read_csv('Evals/sockets.csv', sep=",")]

lmbd = data[0]['lmbd'].tolist()[0:]
yt = list(x/100 for x in range(0, 105, 5))
MAP = [0 for i in range(len(lmbd))]
MRR = [0 for j in range(len(lmbd))]
PK = [0 for k in range(len(lmbd))]


for df in data:
    x = df.iloc[df['AP'].argmax()] #.tolist()[0:]
    print(x)
    y = df['RR'].tolist()[0:]
    z = df['p10'].tolist()[0:]
    MAP = np.add(MAP, x)
    MRR = np.add(MRR, y)
    PK = np.add(PK, z)

MAP = np.divide(MAP, len(data))
MRR = np.divide(MRR, len(data))
PK = np.divide(PK, len(data))

plt.plot(lmbd, MAP, label="MAP", marker='.')
plt.plot(lmbd, MRR, label="MRR", marker='.')
plt.plot(lmbd, PK, label="P@10", marker='.')
plt.xticks(lmbd)
plt.yticks(yt)
plt.xlabel('Lambda Values')
plt.ylabel('Evaluation Metrics')
plt.legend()
plt.savefig("Evals/ALL.png")
plt.show()

