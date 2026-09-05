import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier



def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    X_train = train_df.drop('target', axis=1).values
    y_train = train_df['target'].values
    X_test = test_df.drop('id', axis=1).values
    test_ids = test_df['id'].values
    return X_train, y_train, X_test, test_ids


class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bound_ = Q1 - 1.5 * IQR
        self.upper_bound_ = Q3 + 1.5 * IQR
        return self

    def transform(self, X):
          return np.clip(
            X,
            self.lower_bound_,
            self.upper_bound_
        )


def build_preprocessing_pipeline():
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_clipper', OutlierClipper())
        
    ])


def cross_validate_model(X_train , y_train ):
    skf = StratifiedKFold()
    all_auc = []
    fitted_models = []
    fitted_preprocessors = []
    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
         X_train_fold = X_train[train_index,:]
         x_val_fold = X_train[val_index,:]
         y_train_fold = y_train[train_index]
         y_val_fold = y_train[val_index]
         preprocessor = build_preprocessing_pipeline()
         X_train_fold=preprocessor.fit_transform(X_train_fold)
         x_val_fold=preprocessor.transform(x_val_fold)
         model = XGBClassifier(max_depth=3 , learning_rate = 0.1 , n_estimators=1200 , subsample = 0.8,colsample_bytree = 0.8)
         model.fit(X_train_fold , y_train_fold)
         y_pred = model.predict_proba(x_val_fold)
         auc = roc_auc_score(y_val_fold,y_pred[:,1])
         all_auc.append(auc)
         fitted_models.append(model)
         fitted_preprocessors.append(preprocessor)
         print(f"Fold {i+1} auc : {auc:.3f}")
    return all_auc , fitted_models , fitted_preprocessors

def evaluate_model(X_test , fitted_models ,fitted_preprocessors ):
    all_predictions = []
    for model, preprocessor in zip(fitted_models, fitted_preprocessors):
        X_test_processed  = preprocessor.transform(X_test)
        y_test_pred = model.predict_proba(X_test_processed )
        all_predictions.append(y_test_pred[:,1])
    final_predictions = np.mean(all_predictions, axis=0)
    return final_predictions

X_train , y_train , X_test ,  test_ids = load_data()
auc , fitted_models , fitted_preprocessors = cross_validate_model(X_train,y_train)
mean_auc = np.mean(auc)
print(f"the mean auc : {mean_auc:.3f}")
final_predictions = evaluate_model(X_test , fitted_models ,fitted_preprocessors )
print(f"test pred : {final_predictions}")

# submission_df = pd.DataFrame({'id': test_ids, 'target': final_predictions})
# submission_df.to_csv('submission.csv', index=False)
