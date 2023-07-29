import os
import numpy as np
import subscript

nproc = 32
dname_input = '../data/'
fname_gauge = '../input/gauge.csv'
fname_quake = '../input/quake.csv'

obj = subscript.PreProcess()
print('#####   Initialization   #####')
print('Result => {}'.format(obj.dir))
print('Number of ')
print(' - all scenarios: {}'.format(obj.nsce))
print(' - training scenarios: {}'.format(obj.ntrn))
print(' - testing scenarios: {}'.format(obj.ntst))
print(' - time steps: {}'.format(obj.ntim))
print(' - all gauges: {}'.format(obj.ngag))
print('==============================\n')

#obj.MakeDatamatrix( nproc=nproc, 
#                    dinp=dname_input, 
#                    fgauge=fname_gauge
#                  )
#np.save('../input/Xmat.npy', obj.Xmat)
obj.Xmat = np.load('../input/Xmat.npy')

obj.TTSplit(fquake=fname_quake)
u, s, vh = obj.SVD(X=obj.Xlarn)
os.makedirs(obj.dir+'mode/', exist_ok=True)
np.save(obj.dir+'mode/Ux.npy', u)
np.save(obj.dir+'mode/sx.npy', s)