from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from functools import partial

from .imports import *
from .constants import *

def VectorLookup(dfVectors):
  def vectLookup(input_series):
    indInput = input_series.groupby(INDICIES, sort=False).count().index.to_frame().drop(INDICIES, axis=1)
    dfTsVects = pd.merge(indInput, dfVectors, how='left', left_index=True, right_index=True)
    # display('output_series.shape',dfTsVects.shape)
    return dfTsVects.replace((np.inf, -np.inf), np.NaN).fillna(0)
  return vectLookup

# dfVects = pd.read_feather("/content/drive/My Drive/BP ML data/cached data/dfImuVects-"+"HLV"+".feather").set_index(['file','heartbeat'])
def transform_vectorLookup(dfVects):
  tsfelVectTransform = FunctionTransformer(VectorLookup(dfVects))
  return tsfelVectTransform

def transform_selectFeatures(selected_features):
  def selectFeatures(x, features=[]):
    return x[features]

  vectFeatureSelection = FunctionTransformer(partial(selectFeatures, features=selected_features))
  return vectFeatureSelection


