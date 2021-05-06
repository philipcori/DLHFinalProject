import pandas as pd

path = './explore_version_03/results_whole_dataset_second_stage/resnet152_20200407_multiclass_cv5/result_detail_resnet152_test_cv5.csv'

df = pd.read_csv(path, names=['1', '2', '3', 'A', 'B', 'C', 'D'])

print(df['A'].sum())
print(df['B'].sum())
print(df['C'].sum())
print(df['D'].sum())