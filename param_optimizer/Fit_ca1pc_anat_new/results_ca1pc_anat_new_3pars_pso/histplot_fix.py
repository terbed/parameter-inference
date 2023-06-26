import seaborn as sb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('last.csv')

# sb.jointplot(x='Ra', y='g_pas', data=df, )
df = df.drop('# Fitnes', axis=1)

sb.set_style("darkgrid")
g = sb.pairplot(data=df, diag_kind="kde", markers="+", plot_kws=dict(s=50, edgecolor="b", linewidth=1), diag_kws=dict(shade=True))


for i in range(3):
    g.axes[i, 0].set_xlim((0.5, 2))
    g.axes[i, 0].axvline(x=1., c='g', ls='--', alpha=0.5)
    g.axes[i, 1].set_xlim((0, 400))
    g.axes[i, 1].axvline(x=120., c='g', ls='--', alpha=0.5)
    g.axes[i, 2].set_xlim((1.75e-5, 2.5e-5))
    g.axes[i, 2].axvline(x=0.00002, c='g', ls='--', alpha=0.5)

    #g.axes[0, i].set_ylim((0, 2))
    if i != 0:
        g.axes[0, i].axhline(y=1., c='g', ls='--', alpha=0.5)
    #g.axes[1, i].set_ylim((0, 800))
    if i != 1:
        g.axes[1, i].axhline(y=120., c='g', ls='--', alpha=0.5)
    #g.axes[2, i].set_ylim((0, 4e-4))
    if i != 2:
        g.axes[2, i].axhline(y=0.00002, c='g', ls='--', alpha=0.5)

#g.axes[2, 2].set(xticks=[1e-4, 3e-4])
g.savefig("3p_MLdist_fix.pdf")
g.savefig("3p_MLdist_fix.png")


