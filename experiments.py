from .imports import *
from .constants import *

# https://stackoverflow.com/questions/31259891/put-customized-functions-in-sklearn-pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, ShuffleSplit, TimeSeriesSplit, GroupKFold


## All split_by_* functions return [{'train':<indices for train set>, 'test':<indices for test set>},...]

## Randomly assign samples to train & test set
def split_by_random(dfImu, dfBp, indices=['file', 'heartbeat'], split_kwargs={'n_splits':4, 'test_size':0.2, 'random_state':0}):
  # print('random_split')
  dfAllInds = dfImu.reset_index()[indices].drop_duplicates()
  arrFoldInds = random_split_indices(dfAllInds, split_kwargs)
  return arrFoldInds
  # return get_experiment(arrFoldInds, dfImu, dfBp)

## https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold
def split_by_group(group_col, dfImu, dfBp, indices=['file', 'heartbeat'], split_kwargs={'n_splits':4}):
  dfAllInds = dfImu.reset_index()[indices+[group_col]].drop_duplicates()
  # n_groups = dfAllInds[group_col].nunique() // split_kwargs['n_splits']
  gkf = GroupKFold(n_splits=split_kwargs['n_splits'])
  
  arrFoldInds = []
  for train_index, test_index in gkf.split(dfAllInds, groups=dfAllInds[group_col]):
    arrFoldInds.append({
      'train' : dfAllInds.drop(group_col,axis=1).iloc[train_index],
      'test' : dfAllInds.drop(group_col,axis=1).iloc[test_index]
    })
  return arrFoldInds

# maybe use PredefinedSplit instead?
### fix this fn.. dfAllInds isn't being used. Should use dfTrainInds and dfTestInds to create the train&test indices..
def split_by_query(trainQ, testQ=None, dfBp=None, indices=['file', 'heartbeat'], split_kwargs={'n_splits':4, 'random_state':0}):
  dfAllInds = dfBp.reset_index().drop_duplicates()

  # trainQ = 'index==index' if trainQ=='' else trainQ
  testQ = 'not '+trainQ if testQ is None else testQ
  dfTrain = dfAllInds.query(trainQ).reset_index().set_index(indices)      ## these lines aren't working correctly. Need dfTrainInds to sample from
  dfTest = dfAllInds.query(testQ).reset_index().set_index(indices)      ## these lines aren't working correctly. Need dfTrainInds to sample from

  arrFoldInds = []
  for i in range(split_kwargs['n_splits']):
    train_index = dfTrain.sample(dfTrain.shape[0]//split_kwargs['n_splits']).reset_index()[indices]
    test_index = dfTest.sample(dfTest.shape[0]//split_kwargs['n_splits']).reset_index()[indices]
    arrFoldInds.append({
        'train' : train_index,
        'test' : test_index
    })

  return arrFoldInds

def random_split_indices(dfInds, split_kwargs):
  # print('random_split_indices')
  sss = ShuffleSplit(**split_kwargs)

  arrFoldInds = []
  for train_index, test_index in sss.split(dfInds):
    arrFoldInds.append({
        'train' : dfInds.iloc[train_index],
        'test' : dfInds.iloc[test_index]
    })
  
  return arrFoldInds

def get_experiment(foldInds, dfImu, dfBp):
  # print('get_experiment')
  experimentDfs = {
      'train_x' : get_experiment_df(dfImu, foldInds['train']),
      'train_y' : get_experiment_df(dfBp, foldInds['train']),
      'test_x' : get_experiment_df(dfImu, foldInds['test']),
      'test_y' : get_experiment_df(dfBp, foldInds['test'])
  }
  
  return experimentDfs
  

def get_experiment_df(df, dfInds):
  # print('get_experiment_df', df.reset_index().columns, dfInds.columns)
  return pd.merge(dfInds, df.reset_index(), on=dfInds.columns.to_list(), how='left')


def testPipeline(dfImu, dfBp, pipeline, indices, verbose=False, dropCols=BP_COLS, targetCol='pp', n_splits=5, test_size=0.2, shuffle=True):

    testResults = []
    for experimentIndices in indices:
      objExperimentDfs = get_experiment(experimentIndices, dfImu, dfBp)

      pipeline.fit(objExperimentDfs['train_x'].drop(dropCols, errors='ignore'), objExperimentDfs['train_y'][targetCol])
      preds = pipeline.predict(objExperimentDfs['test_x'])

      y_test = objExperimentDfs['test_y'][targetCol]
      resultsObj = {
          'mean_absolute_error': mean_absolute_error(preds, y_test),
          'r2_score':r2_score(preds, y_test),
          'y_test':y_test,
          'preds':preds
      }
      testResults.append(resultsObj)
      if verbose : print('mean_absolute_error, r2_score : ', round(resultsObj['mean_absolute_error'], 3), round(resultsObj['r2_score'], 3))
        
    return testResults

def getProps(arr, prop):
  return [a[prop] for a in arr]

def resultsToAvgScore(testResults):
  return {
    'mean_absolute_error':np.mean(getProps(testResults,'mean_absolute_error')),
    'r2_score':np.mean(getProps(testResults,'r2_score'))
  }

def resultsToDf(testResults):
  dfRes = pd.concat(getProps(testResults, 'y_test')).to_frame()
  dfRes['preds'] = np.concatenate(getProps(testResults, 'preds'))
  return dfRes
