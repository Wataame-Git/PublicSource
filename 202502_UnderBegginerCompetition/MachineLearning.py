import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
import lightgbm as lgb


#==========================================================================================================================================
#
#==========================================================================================================================================
class MachineLearning():

    #==========================================================================================================================================
    #
    #========================================================================================================================================
    def __init__(self):


        # CSVデータの読み込み 
        self.dfTrainX = pd.read_csv(r'.\data\train.csv', index_col = 0)
        self.dfTrainY = self.dfTrainX["Y"]
        self.dfTrainX = self.dfTrainX.drop(["Y"], axis = 1)

        self.dfTestX = pd.read_csv(r'.\data\test.csv', index_col = 0)


    #==========================================================================================================================================
    #
    #==========================================================================================================================================
    def DataCleansing(self, encording):

        categorical_cols = ["workclass", "education", "marital-status", "occupation",	"relationship", "race",	"sex", "native-country"]
        numerical_cols = ["age", "education-num"]

        # 名義尺度はone-hot-encordingが有用　
        if encording == 'one-hot':

            encoder = OneHotEncoder(sparse_output = False, handle_unknown = "ignore")

            encoded_train = encoder.fit_transform(self.dfTrainX[categorical_cols])
            encoded_test = encoder.transform(self.dfTestX[categorical_cols])  # 訓練データと同じ変換を適用

            # 変換後のカラム名を取得
            encoded_col_names = encoder.get_feature_names_out(categorical_cols)

            # DataFrame に変換
            encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_col_names, index=self.dfTrainX.index)
            encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_col_names, index=self.dfTestX.index)

            # 数値特徴量を統合（One-Hotエンコーディングされたカテゴリ変数と結合）
            self.dfTrainX = pd.concat([self.dfTrainX[numerical_cols], encoded_train_df], axis=1)
            self.dfTestX = pd.concat([self.dfTestX[numerical_cols], encoded_test_df], axis=1)

        # 順序尺度はラベルエンコーディングが有用 
        elif encording == 'label':
        
            # ラベルエンコーディングを一括適用
            label_encoders = {col: LabelEncoder().fit(self.dfTrainX[col]) for col in categorical_cols}
            self.dfTrainX[categorical_cols] = self.dfTrainX[categorical_cols].apply(lambda col: label_encoders[col.name].transform(col))
            self.dfTestX[categorical_cols] = self.dfTestX[categorical_cols].apply(lambda col: label_encoders[col.name].transform(col))

            # 数値特徴量を統合
            self.dfTrainX = self.dfTrainX[numerical_cols + categorical_cols]
            self.dfTestX = self.dfTestX[numerical_cols + categorical_cols]


    #==========================================================================================================================================
    #
    #==========================================================================================================================================
    def SVM(self):
   
        pd.set_option('display.max_rows', None)  # 行の表示を無制限に
        pd.set_option('display.max_columns', None)  # 列の表示を無制限に

        # データの読み込み
        dfTrainX = self.dfTrainX
        dfTestX = self.dfTestX

        # # スケーリング（標準化）
        # scaler = StandardScaler()
        # dfTrainX = scaler.fit_transform(dfTrainX)
        # dfTestX = scaler.transform(dfTestX)

        # 次元削減（PCA）
        pca = PCA(n_components = 5, random_state = 42)
        dfTrainX = pca.fit_transform(dfTrainX)
        dfTestX = pca.transform(dfTestX)

        svm = SVC(random_state = 42)

        # グリッドサーチで調整するハイパーパラメータを定義
        param_grid = {
            'C': [0.1, 1, 10, 100],            # 正則化パラメータ
            'gamma': [1, 0.1, 0.01, 0.001],    # RBFカーネルの係数
            'kernel': ['rbf']        # カーネルの種類
            # 'C': [100],            # 正則化パラメータ
            # 'gamma': [0.1],    # RBFカーネルの係数
            # 'kernel': ['rbf']        # カーネルの種類
        }

        # グリッドサーチを実行し、最適なハイパーパラメータの検索を行う
        grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, cv = 10, scoring = 'accuracy', verbose = 2)
        grid_search.fit(dfTrainX, self.dfTrainY)
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Validation Score:", grid_search.best_score_)

        # 分割した検証用データでの予測
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(dfTestX)

        # # CSV出力
        dfResult = pd.DataFrame(None, index = self.dfTestX.index)
        dfResult["result"] = y_pred
        dfResult.to_csv(r'.\data\result.csv', header=False)

    #==========================================================================================================================================
    #
    #
    #==========================================================================================================================================
    def XGBoost(self):

        dfTrainX = self.dfTrainX
        dfTestX = self.dfTestX

        # # スケーリング（標準化）
        # scaler = StandardScaler()
        # dfTrainX = scaler.fit_transform(dfTrainX)
        # dfTestX = scaler.transform(dfTestX)

        # 次元削減（PCA）
        pca = PCA(n_components = 5, random_state = 42)
        dfTrainX = pca.fit_transform(dfTrainX)
        dfTestX = pca.transform(dfTestX)

        xgb = XGBClassifier(eval_metric = 'logloss')

        # ハイパーパラメータの候補を設定
        param_grid = {
            'n_estimators': [50, 100, 200],   # ブースティングの回数
            'max_depth': [5, 7, 9],          # 決定木の深さ
            'learning_rate': [0.01, 0.1, 0.3], # 学習率
            'subsample': [0.8, 1.0],         # サブサンプルの割合
            'colsample_bytree': [0.8, 1.0]   # 特徴量サブサンプルの割合
        }

        # グリッドサーチ
        grid_search = GridSearchCV(
            estimator = xgb,
            param_grid = param_grid,
            scoring = 'accuracy',  # 評価指標
            cv = 10,                # 交差検証の分割数
            verbose = 1,           # ログ出力
            n_jobs = -1            # 並列実行
        )

        # モデルのトレーニング
        grid_search.fit(dfTrainX, self.dfTrainY)

        # ベストパラメータとスコア
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(dfTestX)

        dfResult = pd.DataFrame(None, index = self.dfTestX.index)
        dfResult["result"] = y_pred

        # CSV出力
        dfResult.to_csv(r'.\data\result.csv', header=False)
        
    #==========================================================================================================================================
    #
    #
    #==========================================================================================================================================
    def RandomForest(self):

        dfTrainX = self.dfTrainX
        dfTestX = self.dfTestX

        # # 標準化　
        # scaler = StandardScaler()
        # dfTrainX = scaler.fit_transform(dfTrainX)
        # dfTestX = scaler.fit_transform(dfTestX)

        model = RFC(random_state=42)

        #　データを分割　
        param_grid = {
            'n_estimators':[50, 100, 200],
            'max_depth':[10, 20, None],
            'min_samples_split':[2, 5, 10],
            'min_samples_leaf':[1, 2, 4],
            'bootstrap':[True, False]
        }

        grid_search = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'accuracy', cv = 10, verbose = 2, n_jobs = -1)
        grid_search.fit(dfTrainX, self.dfTrainY)

        # ベストパラメータとスコア
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(dfTestX)

        dfResult = pd.DataFrame(None, index = self.dfTestX.index)
        dfResult["result"] = y_pred

        # CSV出力
        dfResult.to_csv(r'.\data\result.csv', header=False)

    #==========================================================================================================================================
    # アンサンブル学習
    #
    #==========================================================================================================================================
    def VotingClassfier(self):

        pass




#==========================================================================================================================================
#
#==========================================================================================================================================
if __name__ == '__main__':

    ml = MachineLearning()
    # ml.DataCleansing(encording = 'one-hot')
    ml.DataCleansing(encording = 'one-hot')

    # ml.SVM()
    # ml.XGBoost()
    ml.RandomForest()