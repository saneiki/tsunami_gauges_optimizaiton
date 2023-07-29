import os
import subscript

obj = subscript.Optimization( nmod=21,
                              nopt='ALL',
                              noise_type='unit' #'unit' or 'random'
                            )

GA = True
QR = True
NOWPHAS = True

if GA:
  os.makedirs(obj.dir+'optimization/GA_{}/'.format(obj.fname), exist_ok=True)
  for nopt in obj.nopt:
    print('Number of gauges to be optimized: {}'.format(nopt))
    list_opt, fitness = obj.GAOptimization( nopt=nopt,
                                            nelit=20,
                                            npop=100,
                                            prob_crs=0.9,
                                            prob_mut=0.1
                                          )
    with open(obj.dir \
              + 'optimization/GA_{}/'.format(obj.fname) \
              + '{0:03d}modes_'.format(obj.nmod) \
              + '{0:03d}gauges.csv'.format(nopt), 'w'
              ) as o:
      print(*list_opt, sep=',',file=o)
      print(fitness, file=o)
    print('Optimal set of gauges: {}'.format(list_opt))

if QR:
  os.makedirs(obj.dir+'optimization/QR_{}/'.format(obj.fname), exist_ok=True)
  for nopt in obj.nopt:
    print('Number of gauges to be optimized: {}'.format(nopt))
    list_opt, list_l2norm = obj.QROptimization( nopt=nopt )
    with open(obj.dir \
              + 'optimization/QR_{}/'.format(obj.fname) \
              + '{0:03d}modes_'.format(obj.nmod) \
              + '{0:03d}gauges.csv'.format(nopt), 'w'
              ) as o:
      print(*list_opt, sep=',',file=o)
      for row in list_l2norm:
        print(*row, sep=',', file=o)
    print('Optimal set of gauges: {}'.format(list_opt))

if NOWPHAS:  
  os.makedirs(obj.dir+'optimization/NOWPHAS_{}/'.format(obj.fname), exist_ok=True)
  for nopt in obj.nopt:
    print('Number of gauges to be optimized: {}'.format(nopt))
    list_opt, list_l2norm = obj.NOWPHASOptimization( nopt=nopt )
    with open(obj.dir \
              + 'optimization/NOWPHAS_{}/'.format(obj.fname) \
              + '{0:03d}modes_'.format(obj.nmod) \
              + '{0:03d}gauges.csv'.format(nopt), 'w'
              ) as o:
      print(*list_opt, sep=',',file=o)
      for row in list_l2norm:
        print(*row, sep=',', file=o)
    print('Optimal set of gauges: {}'.format(list_opt))