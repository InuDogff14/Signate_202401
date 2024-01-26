import pandas as pd
from datetime import datetime
import numpy as np
import re

def clean_currency(column):
    return column.str.replace(r'[^0-9.]', '', regex=True).astype(float)


def preprocess(train, test):
    train["train"] = True
    test["train"] = False
    test["MIS_Status"] = None

    data = pd.concat([
        train, test
    ]).reset_index(drop=True)

    data.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    
    data['Sector'] = data['Sector'].astype('str')
    data['Sector'] = data['Sector'].map({
    '11': 'Ag/For/Fish/Hunt',
    '21': 'Min/Quar/Oil_Gas_ext',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'Wholesale_trade',
    '44': 'Retail_trade',
    '45': 'Retail_trade',
    '48': 'Trans/Ware',
    '49': 'Trans/Ware',
    '51': 'Information',
    '52': 'Finance/Insurance',
    '53': 'RE/Rental/Lease',
    '54': 'Prof/Science/Tech',
    '55': 'Mgmt_comp',
    '56': 'Admin_sup/Waste_Mgmt_Rem',
    '61': 'Educational',
    '62': 'Healthcare/Social_assist',
    '71': 'Arts/Entertain/Rec',
    '72': 'Accom/Food_serv',
    '81': 'Other_no_pub',
    '92': 'Public_Admin',
    '0': 'Other',
    })
    
    data.loc[(data['NewExist'] == 1), 'NewBusiness'] = 0
    data.loc[(data['NewExist'] == 2), 'NewBusiness'] = 1
    
    # 'DisbursementDate'列を日付形式に変換
    data['DisbursementDate'] = pd.to_datetime(data['DisbursementDate'], format='%d-%b-%y')
    data['DisbursementYear'] = data['DisbursementDate'].dt.year
    data['DisbursementMonth'] = data['DisbursementDate'].dt.month
    data['DisbursementDay'] = data['DisbursementDate'].dt.day

    data['ApprovalDate'] = pd.to_datetime(data['ApprovalDate'], format='%d-%b-%y')
    data['ApprovalMonth'] = data['ApprovalDate'].dt.month
    data['ApprovalDay'] = data['ApprovalDate'].dt.day

    

    # Applying the function to the specified columns
    data['DisbursementGross'] = clean_currency(data['DisbursementGross'])
    data['GrAppv'] = clean_currency(data['GrAppv'])
    data['SBA_Appv'] = clean_currency(data['SBA_Appv'])

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category')
    return data


def base_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    data = preprocess(train, test)
    return data
