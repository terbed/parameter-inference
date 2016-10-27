import numpy as np
from matplotlib import pyplot

def KullbackLeiber(P,Q):
    KDL = 0
    for i in range(len(P)):
        KDL += P[i]*np.log(P[i]/Q[i])

    return KDL

Ps = raw_input('Posterior probability distribution pathname: ')
Qs = raw_input('Prior probebility distribution pathname: ')

Q = [row[1] for row in np.loadtxt(Qs)]
P = [row[1] for row in np.loadtxt(Ps)]
values = [row[0] for row in np.loadtxt(Qs)]

KDL = KullbackLeiber(P,Q)
print KDL

pyplot.figure()
pyplot.title(Ps + '(r); ' + Qs + '(g)\nKDL: ' + str(KDL))
pyplot.xlabel('probability')
pyplot.ylabel('cm')
pyplot.plot(values, Q, color = 'g')
pyplot.plot(values, P, color = 'r')
pyplot.savefig('KLtestResults/' + Ps + ' ' + Qs + '.png')
pyplot.show()
