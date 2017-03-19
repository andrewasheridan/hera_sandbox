#plots configuration of antenna array
import aipy as a, numpy as n, pylab as p, ephem as e
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import matplotlib.ticker as tic
import seaborn as sns

FILE350 = "../calfiles/HERA_antconfig/antenna_positions_350.dat"
FILE243 = "../calfiles/HERA_antconfig/antenna_positions_243.dat"
FILE128 = "../calfiles/HERA_antconfig/antenna_positions_128.dat"
FILE37 = "../calfiles/HERA_antconfig/antenna_positions_37.dat"

def get_XYI(file):
	antpos = np.loadtxt(file)
	nants = antpos.shape[0]
	X,Y,Z = antpos[:,0], antpos[:,1], antpos[:,2]
	X -= n.average(X)
	Y -= n.average(Y)
	I = n.arange(nants)
	return X,Y,I

sns.set(style="darkgrid")
sns.set_context("poster")
fig, axes = p.subplots(2,2)
axes = axes.flat
names = ['Hera37', 'Hera128', 'Hera243', 'Hera350']
for i, file in enumerate([FILE37, FILE128, FILE243, FILE350]):
	ax = axes[i]
	X, Y, I = get_XYI(file)
	g = min(320, I.shape[0])
	Xg, Yg, Ig = X[:g], Y[:g], I[:g]
	ax.scatter(Xg,Yg)
	# for x,y,i in zip(Xg, Yg,Ig):
	# 	ax.annotate('%s' %i, xy=(x,y), textcoords='data', fontsize=12) # <--
	ax.set_xlabel('East Position [m]')
	ax.set_ylabel('North Position [m]')
	ax.set_aspect(1)
	ax.text(0.04, 0.9, names[i], size=16, transform=ax.transAxes, style='italic', weight='bold')


p.tight_layout()
p.show()
