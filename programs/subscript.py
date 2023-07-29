import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np

import re
import glob
import math
import random
import pandas as pd
import scipy as sp
from scipy import linalg
import dask.array as da
import inverse_problem as IP
from multiprocessing import Pool

class PreProcess:
  def __init__( self, dirs='./result_/', 
                num_sce=1564, num_trn=1414, num_tst=150, 
                num_timestep=4320, num_gag=134
              ):
    os.makedirs(dirs, exist_ok=True)
    self.dir = dirs
    self.nsce = num_sce
    self.ntrn = num_trn
    self.ntst = num_tst
    self.ntim = num_timestep
    self.ngag = num_gag


  def read(self, dt):
    path, gname = dt
    allfiles = []
    for gn in gname:
      allfiles.append('{}/{}.asc'.format(path, gn))
    return np.vstack(
      [np.loadtxt(
      file, delimiter=',', skiprows=1, usecols=1
      ) for file in allfiles]
    )

  
  def MakeDatamatrix(self, nproc, dinp, fgauge):
    gauge = pd.read_csv(fgauge)
    alldirs = sorted(glob.glob(dinp+'JNan_*', recursive=True))
    list_fdat = []
    for dirs in alldirs:
      list_fdat.append([dirs, list(gauge['Obs. Point'])])
    p = Pool(nproc)
    self.Xmat = np.concatenate(p.map(self.read, list_fdat), axis=1)

  
  def TTSplit(self, fquake, seed=7): 
    quake = pd.read_csv(fquake)
    self.cases = pd.DataFrame(quake['ID'])
    self.cases['label'] = 0

    random.seed(seed)
    lst_all = [i for i in range(self.nsce)]
    lst_test = sorted(random.sample(lst_all, int(self.ntst))) 
    lst_larn = [i for i in lst_all if not i in lst_test]

    self.Xlarn = np.zeros((self.Xmat.shape[0], self.ntim*len(lst_larn)))
    self.Xtest = np.zeros((self.Xmat.shape[0], self.ntim*len(lst_test)))
    for i in range(len(lst_larn)):
      self.Xlarn[:,self.ntim*i:self.ntim*(i+1)] = \
        self.Xmat[:,self.ntim*lst_larn[i]:self.ntim*(lst_larn[i]+1)]
    for i in range(len(lst_test)):
      self.Xtest[:,self.ntim*i:self.ntim*(i+1)] = \
        self.Xmat[:,self.ntim*lst_test[i]:self.ntim*(lst_test[i]+1)]
    del self.Xmat
    np.save(self.dir + 'Xlarn.npy', self.Xlarn)
    np.save(self.dir + 'Xtest.npy', self.Xtest)
    
    self.cases.loc[lst_test, 'label'] = 'test'
    self.cases.loc[lst_larn, 'label'] = 'larn'

    self.cases.to_csv(self.dir + 'cases.csv', index=False)

  
  def SVD(self, X):
    X = da.from_array(X.T)
    X = X.rechunk({0:'auto', 1:-1})
    U, S, Vh = da.linalg.tsqr(X, compute_svd=True)
    del X
    U, S, Vh = da.compute(U, S, Vh)
    [U, Vh] = [Vh.T, U.T]
    return U, S, Vh


class Optimization(PreProcess):
  def __init__( self, nmod, nopt, noise_type,
                fname_mode='./result_/mode/Ux.npy'
              ):
    super().__init__()
    self.nmod = nmod
    self.Phi = np.load(fname_mode)
    
    if nopt=='ALL':
      self.nopt = [i for i in range(1, self.ngag, 1)]
    else:
      self.nopt = [nopt]
    
    self.fname = noise_type
    if noise_type=='unit':
      self.sigma = np.eye(self.ngag)
    elif noise_type=='random':
      np.random.seed(0)
      cov = np.random.rand(self.ngag)
      cov[cov<0.5] = 0.01 
      cov[cov>=0.5] = 1
      self.sigma = np.diag(cov)
  
  
  def GAOptimization(self, nopt, nelit, npop, prob_crs, prob_mut, sed=64):
    from deap import base
    from deap import creator
    from deap import tools

    def ObjectFunc(individual, Phi, ngag_opt, nmod, sigma):
      loc = individual 
      if len(loc) != len(set(loc)):
        object = 0.
      else:
        sigma_ind = sigma[loc,:][:,loc]
        if ngag_opt<=nmod:
          object = np.abs(np.linalg.det(
            np.linalg.inv(sigma_ind) @ Phi[loc,:] @ Phi[loc,:].T
          ))
        else:
          object = np.abs(np.linalg.det(
            Phi[loc,:].T @ np.linalg.inv(sigma_ind) @ Phi[loc,:]
          ))
      return object,  

    def MakeIndividual(container, ngag_total, ngag_opt):
      loc = sorted(random.sample(range(ngag_total), k=ngag_opt))
      return container(loc)

    Phi = self.Phi[:,:self.nmod]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", MakeIndividual, creator.Individual, ngag_total=self.ngag, ngag_opt=nopt)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxUniform, indpb=0.9)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.ngag-1, indpb=0.1)
    toolbox.register("evaluate", ObjectFunc, Phi=Phi, ngag_opt=nopt, nmod=self.nmod, sigma=self.sigma)

    random.seed(sed)
    G = 0
    COUNT = 0 
    pop = toolbox.population(n=npop)
    best_ind_prev = tools.selBest(pop, 1)[0]
    better_ind_prev = tools.selBest(pop, nelit)

    for individual in pop:
      individual.fitness.value = toolbox.evaluate(individual)
    while COUNT < 2000:
      G += 1
      offspring = toolbox.select(pop, len(pop)-nelit)
      offspring = list(map(toolbox.clone, offspring))
      for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < prob_crs:
          toolbox.mate(child1, child2)
          del child1.fitness.values
          del child2.fitness.values
      for mutant in offspring:
        if random.random() < prob_mut:
          toolbox.mutate(mutant)
          del mutant.fitness.values        
      invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
      fitnesses = map(toolbox.evaluate, invalid_ind)
      for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
      pop[:] = offspring
      pop += better_ind_prev
      best_ind = tools.selBest(pop, 1)[0]
      better_ind_prev = tools.selBest(pop, nelit)
      if best_ind == best_ind_prev:
        COUNT += 1
      else:
        COUNT = 0
      best_ind_prev = best_ind
    del creator.FitnessMax
    del creator.Individual
    del toolbox
    return best_ind, best_ind.fitness.values[0]
  
  
  def MakeListl2norm(self, l2list, dmat, list_gauge):
    l2norm = np.sqrt(np.sum(dmat**2, axis=0))
    l2norm = l2norm.reindex(index=list_gauge)
    l2norm = l2norm.fillna(0)
    l2list.append(list(l2norm))
    return l2list
  
  
  def Householder(self, Hmat, itern, vector):
    Hmat_sub  = np.eye(np.size(vector)) \
              - (2*vector@vector.T)/np.sum(vector**2)
    Hmat[itern:, itern:] = Hmat_sub 
    return Hmat

  
  def ColumnPivoting(self, itern, dmat, l2norm):
    list_column = list(dmat.columns)
    list_column.remove(l2norm.idxmax())
    list_column.insert(itern, l2norm.idxmax())
    dmat = dmat.reindex(columns=list_column) 
    return dmat


  def QROptimization(self, nopt): 
    Phi = self.Phi[:,:self.nmod]

    idx = [ 'index{}'.format(i+1) for i in range(self.ngag) ]
    col = [ 'gauge{}'.format(i+1) for i in range(self.ngag) ]

    Mobj = sp.linalg.sqrtm(
      np.linalg.inv(self.sigma)
    ) @ Phi @ Phi.T @ sp.linalg.sqrtm(
      np.linalg.inv(self.sigma)
    ).T
    Bmat = pd.DataFrame(Mobj, columns=col[:Mobj.shape[1]], index=idx[:Mobj.shape[0]])
    list_gauge = list(Bmat.columns)
    optlist = []
    l2list = []

    for jj in range(nopt):
      Hmat = np.eye(Mobj.shape[0])
      Bmat_sub = Bmat.iloc[jj:, jj:]
      norm_column = np.sqrt(np.sum((Bmat_sub**2),axis=0)) 
      opt_idx = int(norm_column.idxmax().replace('gauge',''))-1
      optlist.append(opt_idx)
      l2list = self.MakeListl2norm(l2list=l2list, dmat=Bmat_sub, list_gauge=list_gauge)

      avec = np.array(Bmat_sub.loc[:, norm_column.idxmax()]).reshape(-1,1)
      bvec = np.zeros(avec.size).reshape(-1,1)
      bvec[0] = norm_column.max()
      Hmat = self.Householder(Hmat=Hmat, itern=jj, vector=avec-bvec)
      Bmat = self.ColumnPivoting(itern=jj, dmat=Bmat, l2norm=norm_column)
      Bmat = Hmat @ Bmat
    
    return optlist, l2list

  
  def NOWPHASOptimization(self, nopt):
    Phi = self.Phi[:,:self.nmod]
    list_nowphas = [108, 107, 133, 125, 129, 71, 92] 

    idx = [ 'index{}'.format(i+1) for i in range(self.ngag) ]
    col = [ 'gauge{}'.format(i+1) for i in range(self.ngag) ]

    Mobj = sp.linalg.sqrtm(
      np.linalg.inv(self.sigma)
    ) @ Phi @ Phi.T @ sp.linalg.sqrtm(
      np.linalg.inv(self.sigma)
    ).T
    Bmat = pd.DataFrame(Mobj, columns=col[:Mobj.shape[1]], index=idx[:Mobj.shape[0]])
    list_gauge = list(Bmat.columns)
    optlist = []
    l2list = []

    for jj in range(nopt): 
      Hmat = np.eye(Mobj.shape[0])
      Bmat_sub = Bmat.iloc[jj:, jj:]
      norm_column = np.sqrt(np.sum((Bmat_sub**2),axis=0)) 

      if jj<len(list_nowphas):
        optlist.append(list_nowphas[jj])
        avec = np.array(
          Bmat_sub.loc[:, 'gauge{}'.format(list_nowphas[jj]+1)]
        ).reshape(-1,1)
        bvec = np.zeros(avec.size).reshape(-1,1)
        bvec[0] = np.sqrt(np.sum(avec**2))
        Hmat = self.Householder(Hmat=Hmat, itern=jj, vector=avec-bvec)
        list_column = list(Bmat.columns)
        list_column.remove('gauge{}'.format(list_nowphas[jj]+1))
        list_column.insert(jj, 'gauge{}'.format(list_nowphas[jj]+1))
        Bmat = Bmat.reindex(columns=list_column) 
        Bmat = Hmat @ Bmat
      else:
        opt_idx = int(norm_column.idxmax().replace('gauge',''))-1
        optlist.append(opt_idx)
        avec = np.array(Bmat_sub.loc[:, norm_column.idxmax()]).reshape(-1,1)
        bvec = np.zeros(avec.size).reshape(-1,1)
        bvec[0] = norm_column.max()
        Hmat = self.Householder(Hmat=Hmat, itern=jj, vector=avec-bvec)
        Bmat = self.ColumnPivoting(itern=jj, dmat=Bmat, l2norm=norm_column)
        Bmat = Hmat @ Bmat
      l2list = self.MakeListl2norm(l2list=l2list, dmat=Bmat_sub, list_gauge=list_gauge)
    return optlist, l2list


class PSR(PreProcess):
  def __init__( self, nmod, 
                fname_mode='./result_/mode/Ux.npy'
              ):
    super().__init__()
    self.nmod = nmod
    self.Phi = np.load(fname_mode)
  
  def Waveform(self, om, nt, no, ts=135, solver='Kalman'):
    os.makedirs(self.dir+'SuperResolution/Waveform/', exist_ok=True)
    if om=='RANDOM':
      random.seed(1)
      optlist = random.sample([i for i in range(self.ngag)], no)
    else:
      optlist = self.ReadFirstRow(om=om, nt=nt, no=no)
    Xtest = np.load(self.dir + 'Xtest.npy')[optlist, 
                                            ts*self.ntim:(ts+1)*self.ntim
                                            ]
    Alpha = eval('IP.' + solver + 'Inverse')(
      Xobs=Xtest, Phi=self.Phi, rnum=self.nmod, optlist=optlist
    )
    Xrecon = self.Phi[:,:self.nmod] @ Alpha
    names = ['gauge No.{}'.format(i+1) for i in range(self.ngag)]
    df = pd.DataFrame(data=Xrecon, index=names)
    df.to_csv(self.dir \
              +'SuperResolution/Waveform/{}_{}_{}sce_{}opt.csv'
              .format(om, solver, ts, no)
              )
  
  def para(self, dat):
    p, Xtest, optlist = dat
    error_squre = 0
    for it in range(int(Xtest.shape[1]/self.ntim)):
      Xobs=Xtest[optlist,it*self.ntim:(it+1)*self.ntim]
      Alpha = IP.KalmanInverse(
        Xobs=Xobs, Phi=self.Phi, rnum=self.nmod, optlist=optlist
      )
      Xrecon = self.Phi[:,:self.nmod] @ Alpha
      error_squre += np.average((Xtest[:,it*self.ntim:(it+1)*self.ntim] - Xrecon)**2)
    return error_squre
  
  
  def WholeSRError_off(self, Xtest, optlist, solver, cov, proc=50):
    if solver=='Kalman':
      Xtest_proc = []
      for p in range(proc):
        ini = math.floor(self.ntst * p / proc)
        fin = math.floor(self.ntst * (p+1) / proc)
        Xtest_proc.append([ p, 
                            Xtest[:,ini*self.ntim:fin*self.ntim], 
                            optlist
                          ])
      with Pool(processes=proc) as p:
        result_list = p.map(func=self.para, iterable=Xtest_proc)
      rmse = np.sqrt(np.sum(result_list)/self.ntst)

    elif solver=='Pseudo':
      Alpha = IP.PseudoInverse(
        Xobs=Xtest[optlist,:], Phi=self.Phi, rnum=self.nmod, optlist=optlist
      )
      Xrecon = self.Phi[:,:self.nmod] @ Alpha
      rmse = np.sqrt(np.average((Xtest - Xrecon)**2))
    return rmse
  
  
  def para_on(self, dat):
    p, Xtest, Xobs_all, optlist = dat
    error_squre = 0
    for it in range(int(Xtest.shape[1]/self.ntim)):
      Xobs=Xobs_all[optlist,it*self.ntim:(it+1)*self.ntim]
      Alpha = IP.KalmanInverse(
        Xobs=Xobs, Phi=self.Phi, rnum=self.nmod, optlist=optlist
      )
      Xrecon = self.Phi[:,:self.nmod] @ Alpha
      error_squre += np.average((Xtest[:,it*self.ntim:(it+1)*self.ntim] - Xrecon)**2)
    return error_squre
  
  
  def WholeSRError_on(self, Xtest, optlist, solver, cov, proc=50):
    np.random.seed(1000)
    Xnoise = Xtest + np.random.multivariate_normal(
      mean=np.zeros(Xtest.shape[0]), cov=cov, size=np.shape(Xtest)[1]
    ).T

    if solver=='Kalman':
      Xtest_proc = []
      for p in range(proc):
        ini = math.floor(self.ntst * p / proc)
        fin = math.floor(self.ntst * (p+1) / proc)
        Xtest_proc.append([ p,
                            Xtest[:,ini*self.ntim:fin*self.ntim],
                            Xnoise[:,ini*self.ntim:fin*self.ntim],
                            optlist])
      with Pool(processes=proc) as p:
        result_list = p.map(func=self.para_on, iterable=Xtest_proc)
      rmse = np.sqrt(np.sum(result_list)/self.ntst)

    elif solver=='Pseudo':
      Alpha = IP.PseudoInverse(
        Xobs=Xtest[optlist,:], Phi=self.Phi, rnum=self.nmod, optlist=optlist
      )
      Xrecon = self.Phi[:,:self.nmod] @ Alpha
      rmse = np.sqrt(np.average((Xtest - Xrecon)**2))
    return rmse
  
  
  def SRErrorVsGauge(self, om, nt, tn, solver='Kalman'):
    os.makedirs(self.dir+'SuperResolution/WholeError/noise_{}/'.format(
      tn), exist_ok=True)
    fname = self.dir \
          + 'SuperResolution/WholeError/noise_{}/{}_{}.csv'.format(
          tn, om+nt, solver
          )
    print(' Result => {}'.format(fname))
    f = open(fname, 'w')
    f.writelines(['Number of optimal gauges', ',', 'RMSE', '\n'])
    f.close()
    
    self.fname = nt
    if nt=='unit':
      self.sigma = np.eye(self.ngag)
    elif nt=='random':
      np.random.seed(0)
      cov = np.random.rand(self.ngag)
      cov[cov<0.5] = 0.01 
      cov[cov>=0.5] = 1
      self.sigma = np.diag(cov)
    
    Xtest = np.load(self.dir + 'Xtest.npy')
    list_num_opt = [i for i in range(1, self.ngag)]
    for nopt in list_num_opt:
      if om=='RANDOM':
        trials=100
        rmse_whole = 0
        for itr in range(trials):
          random.seed(itr)
          optlist = random.sample([i for i in range(self.ngag)], nopt)
          
          rmse_whole += self.WholeSRError_off(
            Xtest=Xtest, optlist=optlist, solver=solver, cov=self.sigma
          )
        rmse_whole /= trials
      else:
        if om=='NOWPHAS' and nopt<7:
          continue
        optlist = self.ReadFirstRow(om=om, nt=nt, no=nopt)
        rmse_whole = eval('self.WholeSRError_' + tn)(
            Xtest=Xtest, optlist=optlist, solver=solver, cov=self.sigma)
      f = open(fname, 'a')
      f.writelines([str(nopt), ',', str(rmse_whole), '\n'])
      f.close()

  
  def ReadFirstRow(self, om, nt, no):
    input_data = open(self.dir \
                      + 'optimization/{}_{}/'.format(om, nt) \
                      + '{0:03d}modes_'.format(self.nmod) \
                      + '{0:03d}gauges.csv'.format(no), 'r'
                      )
    for row in input_data:
      if not re.match('#', row):
        first_row = row.rstrip('\n').split(',')
        break
    input_data.close()  
    return [int(i) for i in first_row]
