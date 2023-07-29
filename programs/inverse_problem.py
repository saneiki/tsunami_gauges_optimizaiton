import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
import numpy as np

def PseudoInverse(Xobs, Phi, rnum, optlist):
  Alpha_est = np.linalg.pinv(Phi[optlist,:rnum]) @ Xobs
  return Alpha_est

def KalmanInverse(Xobs, Phi, rnum, optlist):
  avec = np.zeros(rnum)
  Pmat = np.eye(rnum)
  Sigma_w = np.eye(len(optlist))
  Sigma_v = 0.01*np.eye(rnum)
  Alpha_est = np.zeros(( rnum, Xobs.shape[1] ))
  for itime in range(Alpha_est.shape[1]):
    a_bar = avec
    P_bar = Pmat + Sigma_v
    avec = a_bar \
      + P_bar @ Phi[optlist,:rnum].T @ np.linalg.inv(Sigma_w) \
      @ (Xobs[:,itime] - Phi[optlist,:rnum] @ a_bar)
    Pmat = np.linalg.inv( np.linalg.inv(P_bar) \
      + Phi[optlist,:rnum].T @ np.linalg.inv(Sigma_w) \
      @ Phi[optlist,:rnum])
    Alpha_est[:,itime] = avec
  return Alpha_est