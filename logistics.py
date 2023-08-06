from statistics import mode
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

csv=pd.read_csv('P2_consignment_data.csv')
csv_1=pd.read_csv('P2_consignment_volume_data.csv')
df_1=pd.DataFrame(csv_1)
df=pd.DataFrame(csv)
df_1.drop(axis=1,labels='created_at',inplace=True)
df_1.loc[df_1['unit']=='IN',['length','breadth','height']]*=2.54
df_1.loc[df_1['unit']=='IN','unit']='CM'
df.drop(axis=1,labels=['cnote','created_date','delivered','Delivery_Date','QC_Validation','cpm'],inplace=True);df.rename(inplace=True,columns={'id':'consignment_id','Voloume':'Volume'})
df['density']=df['weight']/df['Volume']
df=df.merge(right=df_1,on='consignment_id',how='outer')
df_groped=df.groupby(by=['client_id']).agg(
    total_weight=('weight','sum'),
    total_volume=('Volume','sum'),
    total_no_of_consigment=('consignment_id','count'),
    total_no_of_boxes=('total_boxes','sum'),
    Mean_CFT=('density','mean'),
    CFT_Percentile_25_CFT=('density',lambda x:x.quantile(.25)),
    CFT_Percentile_75_CFT=('density',lambda x:x.quantile(.75))
    )

df_groped['CFT_IQR']=df_groped['CFT_Percentile_75_CFT']-df_groped['CFT_Percentile_25_CFT']
df_groped['CFT_lower_limit']=df_groped['CFT_Percentile_25_CFT']-(df_groped['CFT_IQR']*1.5)
df_groped['CFT_upper_limit']=df_groped['CFT_Percentile_75_CFT']+(df_groped['CFT_IQR']*1.5)
# pd.set_option('display.max_columns', None)
merged_df=df_groped.merge(right=df,on='client_id',how='outer')


def sorting(row):
    sorted_dim=sorted([row['length'],row['breadth'],row['height']])
    row['length']=sorted_dim[2]
    row['breadth']=sorted_dim[1]
    row['height']=sorted_dim[0]
    return row

df=df.apply(sorting,axis=1)

dimention_table=df.groupby(by='client_id').agg(
    Max_length=('length','max'),
    Most_frequent_length=('length', lambda x: mode(x)),
    freq_of_most_freq_length=('length',lambda x: x.tolist().count(mode(x))),
    freq_of_max_length=('length', lambda x: (x==x.max()).sum()),
    percentile_25_length=('length',lambda x: x.quantile(.25)),
    percentile_75_length=('length',lambda x: x.quantile(.75)),
)
dimention_table['IQR_length']=dimention_table['percentile_75_length']-dimention_table['percentile_25_length']
dimention_table['Length_IQR_lower_limit']=dimention_table['percentile_25_length']-(dimention_table['IQR_length']*1.5)
dimention_table['Length_IQR_upper_limit']=dimention_table['percentile_75_length']+(dimention_table['IQR_length']*1.5)
# print(dimention_table.head())

weight_table=df.groupby(by='client_id').agg(
    Max_weight=('weight','max'),
    Most_frequent_weight=('weight', lambda x: mode(x)),
    freq_of_most_freq_weight=('weight',lambda x: x.tolist().count(mode(x))),
    freq_of_max_weight=('weight', lambda x: (x==x.max()).sum()),
    percentile_25_weight=('weight',lambda x: x.quantile(.25)),
    percentile_75_weight=('weight',lambda x: x.quantile(.75)),
)
weight_table['IQR_weight']=weight_table['percentile_75_weight']-weight_table['percentile_25_weight']
weight_table['weight_IQR_lower_limit']=weight_table['percentile_25_weight']-(weight_table['IQR_weight']*1.5)
weight_table['weight_IQR_upper_limit']=weight_table['percentile_75_weight']+(weight_table['IQR_weight']*1.5)
# print(weight_table.head())
# print(merged_df.head())
merged_df=merged_df.merge(right=dimention_table,on='client_id',how='outer')
merged_df=merged_df.merge(right=weight_table,on='client_id',how='outer')
# print(merged_df.columns)

def outliers(fd):
    fd['CFT_Outlier']=(fd['density']>fd['CFT_upper_limit']) | (fd['density']<fd['CFT_lower_limit'])
    fd['Length_Outlier']=(fd['length']>fd['Length_IQR_upper_limit']) | (fd['length']<fd['Length_IQR_lower_limit'])
    fd['weight_Outlier']=(fd['weight']>fd['weight_IQR_upper_limit']) | (fd['weight']<fd['weight_IQR_lower_limit'])
    fd['fill_Outlier']=(fd['length']==1) & (fd['breadth']==1) & (fd['height']==1)
    return fd

merged_df=outliers(merged_df)

CFT_outlier=merged_df['CFT_Outlier'].sum()/len(merged_df)*100
Length_outlier=merged_df['Length_Outlier'].sum()/len(merged_df)*100
weight_outlier=merged_df['weight_Outlier'].sum()/len(merged_df)*100
fagged_outlier=merged_df['fill_Outlier'].sum()/len(merged_df)*100
# print(CFT_outlier,Length_outlier,weight_outlier,fagged_outlier)

result=merged_df.groupby(by=['client_id','industry_type']).agg(
    Total_consignment=('consignment_id','count'),
    Total_CFT_outlier=('CFT_Outlier','sum'),
    Total_Length_outlier=('Length_Outlier','sum'),
    Total_weigth_outlier=('weight_Outlier','sum'),
    Total_Flagged_outlier=('fill_Outlier','sum'),
)
result['CFT_outlier_percentage']=result['Total_CFT_outlier']/result['Total_consignment']*100
result['Length_outlier_percentage']=result['Total_Length_outlier']/result['Total_consignment']*100
result['weight_outlier_percentage']=result['Total_weigth_outlier']/result['Total_consignment']*100
result['Flagged_outlier_percentage']=result['Total_Flagged_outlier']/result['Total_consignment']*100

# print(result.head())
print(merged_df.columns)


##--------------------------------------------------------------------------------------------------------
merged_df['wrong_observation']=merged_df['CFT_Outlier'] | merged_df['Length_Outlier'] | merged_df['weight_Outlier'] | merged_df['fill_Outlier']
# print(merged_df.head())