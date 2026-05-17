import pandas as pd
df = pd.read_csv('diabetic_data.csv')
#print(df.shape)
#print(df['readmitted'].value_counts())
#print(df.isnull().sum())
#print(df.dtypes)
#print(df['max_glu_serum'].value_counts())
#print(df['A1Cresult'].value_counts())
print(df['gender'].value_counts())