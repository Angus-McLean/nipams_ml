from .imports import *
from .constants import *

from scipy.io import loadmat
import glob
from google.colab import drive

## Connect & Authenticate to Google Drive
def connect():
  drive.mount('/content/drive')

## Given a dictImuData object, convert the data to pandas dataframes.
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

def getContinuousDataFromMatlab(dictImuData):
  dfBpRaw = pd.concat([
    pd.DataFrame(dictImuData['bio_data'][:,[17,14,15]], columns=BP_COLS),
    pd.DataFrame(dictImuData['bio_ts']).T.rename(columns={0:'ts'}),
    pd.DataFrame(dictImuData['bio_data'][:,12], columns=['ecgTs'])
  ], axis=1)
  dfBpRaw.ts = pd.to_timedelta(dfBpRaw.ts, unit='s')

  dfImuRaw = pd.concat([
    pd.DataFrame(dictImuData['imu_data'], columns=IMU_DATA_COLS),
    pd.DataFrame(dictImuData['imu_ts']).rename(columns={0:'ts'})
  ], axis=1)
  dfImuRaw.ts = pd.to_timedelta(dfImuRaw.ts, unit='s')

  return dfImuRaw.set_index('ts'), dfBpRaw.set_index('ts')


## Given list of files, load matlab matrix and parse to dataframes
def loadFiles(files, continuous=False):
  arrDfsBp = []
  arrDfsImu = []

  for f in files:
    imuFile = loadmat(f);
    if continuous:
      dfImu, dfBp = getContinuousDataFromMatlab(imuFile)
    else:
      dfBp = getBpDataFromMatlab(imuFile)
      dfImu = getImuDataFromMatlab(imuFile)
    dfBp['file'] = f
    dfImu['file'] = f
    arrDfsBp.append(dfBp)
    arrDfsImu.append(dfImu)

  dfImuAll = pd.concat(arrDfsImu)
  dfBpAll = pd.concat(arrDfsBp)
  if not continuous:
    dfImuAll = dfImuAll.reset_index().set_index(INDICIES + IMU_TS)
    dfBpAll = dfBpAll.reset_index().set_index(INDICIES)
  
  return dfImuAll, dfBpAll


## Load from File Path Patterns
def loadFilePath(filepaths=None, drop_na=True, continuous=False, parseFileName=False):
  ## Given list of fileQueries, iterate all and glob matching files, return file names
  if type(filepaths)==str : filepaths = [filepaths]
  filenames = [glob.glob(n) for n in filepaths]
  filenames = [item for sublist in filenames for item in sublist]

  arrFilenameSubset = pd.Series(filenames).to_list()
  dfImuAll, dfBpAll = loadFiles(arrFilenameSubset, continuous=continuous)
  
  if drop_na :
    dfImu = dfImuAll.dropna() #.head(1000*500)    ### TODO : REMOVE THE "HEAD" call
    dfBp = dfImu.groupby(INDICIES).first().merge(dfBpAll, on=INDICIES, how='left')[dfBpAll.columns]
    print('dfImu Before & After drop NAs : ', dfImuAll.shape, dfImu.shape, 'Dropped # Rows :', dfImuAll.shape[0] - dfImu.shape[0])
    print('dfBp Before & After drop NAs : ', dfBpAll.shape, dfBp.shape, 'Dropped # Rows :', dfBpAll.shape[0] - dfBp.shape[0])
    dfImuAll = dfImu
    dfBpAll = dfBp
  
  if parseFileName:
    dfImuAll = parseFileInfo(dfImuAll)
    dfBpAll = parseFileInfo(dfBpAll)
  
  dfImuAll[IMU_DATA_COLS] = dfImuAll[IMU_DATA_COLS].astype('float32') #Hello
  return dfImuAll, dfBpAll

## Parse Patient, TestType and TestNum.
def parseFileInfo(df, filenameCol='file'):
  ## TODO : Speed this up by taking only unique filenames, parse those, then do a join on original df
  a = df[filenameCol].str.extract('(sub\d+)_([A-z]+)(\d*)\.mat')
  a.columns = ['patient','test_type','test_num']
  return pd.concat([df, a], axis=1)

## Downsample Signals to appropriate frequency
def interpolateDatasets(dfImu, dfBp, freq='5ms', groupbyCol='file', method='linear'):
  dfBpSamp = dfBp.groupby(groupbyCol).apply(lambda x:x.resample(freq).interpolate(method=method)).reset_index(groupbyCol, drop=True)
  dfImuSamp = dfImu.groupby(groupbyCol).apply(lambda x:x.resample(freq).interpolate(method=method)).reset_index(groupbyCol, drop=True)
  
  # interpolation doesn't work from strings.. fill in the NaNs
  serBpNonNumCols = dfBpSamp.columns[~dfBpSamp.columns.isin(BP_COLS)]
  dfBpSamp[serBpNonNumCols] = dfBpSamp[serBpNonNumCols].fillna(method='ffill')
  serImuNonNumCols = dfImuSamp.columns[~dfImuSamp.columns.isin(IMU_DATA_COLS)]
  dfImuSamp[serImuNonNumCols] = dfImuSamp[serImuNonNumCols].fillna(method='ffill')

  return dfImuSamp, dfBpSamp

## Segment