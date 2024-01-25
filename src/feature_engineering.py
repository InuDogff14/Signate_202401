import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import Feature, generate_features, create_memo
from preprocess import base_data
import numpy as np
import pandas as pd
import hydra
import re
import goto_conversion
from math import radians, cos, sin, asin, sqrt


# 生成された特徴量を保存するパス
Feature.dir = "features"
# trainとtestを結合して基本的な前処理を行ったデータを呼ぶ
data = base_data()


class Base_data(Feature):
    def create_features(self):
        self.data = data.drop(['DisbursementDate','ApprovalDate'], axis=1)
        create_memo("base_data", "初期")


class IsSameState(Feature):
    def create_features(self):
        data['State'] = data['State'].astype(str)
        data['BankState'] = data['BankState'].astype(str)
        isSameState = (data['State'] == data['BankState']).astype('category')
        df = isSameState.to_frame(name='IsSameState')
        self.data = df.copy()
        create_memo("IsSameState", "StateとBankStateが一致するかどうかを示す特徴量を作成")

class SBAGuaranteeRatio(Feature):
    def create_features(self):
        ratio = data['SBA_Appv'] / data['GrAppv']
        df = ratio.to_frame(name = 'SBAGuaranteeRatio')
        self.data = df.copy()
        create_memo("SBAGuaranteeRatio","SBAが保証する金額と承認された総額の比率。これはSBAの保証がローンに占める割合を示します。")

class LoanvsApprovedAmountDifference(Feature):
    def create_features(self):
        diff = data['GrAppv'] - data['DisbursementGross']
        df = diff.to_frame(name="LoanvsApprovedAmountDifference")
        self.data = df.copy()
        create_memo("LoanvsApprovedAmountDifference","融資額と承認額の差。これは実際に支払われた金額と銀行によって承認されたローンの総額の差を表します。")


class LoanProcessingPeriod(Feature):
    def create_features(self):
        period = (data['DisbursementDate'] - data['ApprovalDate']).dt.days
        df = period.to_frame(name="LoanProcessingPeriod")
        self.data = df.copy()
        create_memo("LoanProcessingPeriod","融資処理期間。これは融資の承認日から支払日までの期間（日数）です。")


class YearsSinceApproval(Feature):
    def create_features(self):
        diff = 2024 - data['ApprovalFY']
        df = diff.to_frame(name="YearsSinceApproval")
        self.data = df.copy()
        create_memo("YearsSinceApproval","融資承認からの経過年数。これは現在の年度から承認された財務年度を差し引いた値で、融資が承認されてからどれくらいの時間が経過したかを示します。")


class IsFranchise(Feature):
    def create_features(self):
        isFranchise = data['FranchiseCode'].apply(lambda x: 0 if x == 0 else 1)
        df = pd.DataFrame(isFranchise, columns=['FranchiseFlag'])
        self.data = df.astype('category').copy()
        create_memo("IsFranchise","フランチャイズかどうか")


class JobsCreatedPerEmployee(Feature):
    def create_features(self):
        temp = data['CreateJob'] / data['NoEmp']
        df = temp.to_frame(name='Jobs_Created_per_Employee')
        self.data = df.copy()
        create_memo('Jobs_Created_per_Employee',"Jobs_Createdの従業員比")


class JobsRetainedPerEmployee(Feature):
    def create_features(self):
        temp = data['RetainedJob'] / data['NoEmp']
        df = temp.to_frame(name='Jobs_Retained_per_Employee')
        self.data = df.copy()
        create_memo('Jobs_Retained_per_Employee',"JobsRetainedの従業員比")


class LoanAmountPerEmployee(Feature):
    def create_features(self):
        temp = data['DisbursementGross'] / data['NoEmp']
        df = temp.to_frame(name = 'Loan_Amount_per_Employee')
        self.data = df.copy()
        create_memo('Loan_Amount_per_Employee','以下略')


class CityCount(Feature):
    def create_features(self):
        temp = data.groupby('City')['City'].transform('count')
        df = pd.DataFrame({'CityCount': temp})
        self.data = df.copy()
        create_memo('CityCount', 'CityのCount')


class StateCount(Feature):
    def create_features(self):
        temp = data.groupby('State')['State'].transform('count')
        df = pd.DataFrame({'StateCount': temp})
        self.data = df.copy()
        create_memo('StateCount','StateのCount')


class BankStateCount(Feature):
    def create_features(self):
        temp = data.groupby('BankState')['BankState'].transform('count')
        df = pd.DataFrame({'BankStateCount': temp})
        self.data = df.copy()
        create_memo('BankStateCount','BankStateのCount')


class SectorCount(Feature):
    def create_features(self):
        temp = data.groupby('Sector')['Sector'].transform('count')
        df = pd.DataFrame({'SectorCount': temp})
        self.data = df.copy()
        create_memo('SectorCount','SectorのCount')


class LoanToValueRatio(Feature):
    def create_features(self):
        temp = data['DisbursementGross'] / data['GrAppv']
        df = temp.to_frame(name='LoanToValueRatio')
        self.data = df.copy()
        create_memo('LoanToValueRatio','DisbursementGross（支払総額）とGrAppv（承認額）の比率。これは融資額が承認額に対してどれくらいの比率であるかを示します。')


class BusinessAgeAtDisbursement(Feature):
    def create_features(self):
        # Convert DisbursementDate to datetime
        data['DisbursementDate'] = pd.to_datetime(data['DisbursementDate'], errors='coerce')
        # Assuming new businesses have an age of 0 and existing businesses have an age of 1 or more
        temp = data['NewExist'].apply(lambda x: 0 if x == 1 else 1)
        df = temp.to_
@hydra.main(config_name="../config/config.yaml")
def run(cfg):
    print(cfg)
    # overwriteがfalseなら上書きはされない
    # globals()からこのファイルの中にある特徴量クラスが選別されてそれぞれ実行される
    generate_features(
        globals(),
        cfg['']['']['']['config']['base']['overwrite']
        )


# デバッグ用
if __name__ == "__main__":
    run()
