from .imports import *
from .constants import *

from scipy.io import loadmat
import glob
from google.colab import drive


def connect():
  drive.mount('/content/drive')

def getBpDataFromMatlab(dictImuData):
  bp_all = np.asarray(dictImuData['mlts_now'])
  dfBp = pd.DataFrame(bp_all, columns=BP_COLS)
  dfBp.index.name = 'heartbeat'
  return dfBp

def getImuDataFromMatlab(dictImuData):
  imu_all = np.asarray(dictImuData['mlfs_now'])
  imuShape = imu_all.shape
  imuInd = pd.MultiIndex.from_product([range(imuShape[2]), range(imuShape[0])], names=['heartbeat','step_index'])
  dfRaw = pd.concat([pd.DataFrame(imu_all[:,:,i]) for i in range(imuShape[2])]).reset_index(drop=True)
  dfRaw.columns = IMU_COLS
  dfRaw = dfRaw.set_index(imuInd).reset_index().drop('step_index', axis=1).set_index(['heartbeat','ts'])
  return dfRaw

def toPlotlyDf(df, id_vars=['ts']):
  df = df.reset_index()
  return pd.melt(df, id_vars=id_vars, value_vars=df.drop(id_vars, axis=1).columns)

def loadFiles(files):
  arrDfsBp = []
  arrDfsImu = []

  for f in files:
    imuFile = loadmat(f);
    dfBp = getBpDataFromMatlab(imuFile)
    dfImu = getImuDataFromMatlab(imuFile)
    dfBp['file'] = f
    dfImu['file'] = f
    arrDfsBp.append(dfBp)
    arrDfsImu.append(dfImu)

  dfImuAll = pd.concat(arrDfsImu).reset_index().set_index(['file','heartbeat','ts'])
  dfBpAll = pd.concat(arrDfsBp).reset_index().set_index(['file','heartbeat'])
  return dfImuAll, dfBpAll

def loadFilePath(filepath=None):
  filenames = glob.glob(filepath)
  # print('Found', len(filenames), 'Data Files')

  arrFilenameSubset = pd.Series(filenames).to_list()
  dfImuAll, dfBpAll = loadFiles(arrFilenameSubset)
  
  return dfImuAll.astype('float16'), dfBpAll