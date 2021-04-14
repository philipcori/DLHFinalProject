import pickle
import random
import os
import csv

normal_list = []
non_normal_list_covid = []
non_normal_list_kaggle = []

formal_covid_dict = pickle.load(open('./data_preprocess/normal_non_normal_covid_dict.pkl', 'rb'))

for key, value in formal_covid_dict.items():
    for image_name, info in value['image_dict'].items():
        if 'AP' or 'PA' in info['type']:
            if value['class']['normal'] == '1':
                normal_list.append((info['path'], key + '_' + image_name, 1))
            else:
                non_normal_list_covid.append((info['path'], key + '_' + image_name, 0))

formal_xray_dict = pickle.load(open('./data_preprocess/normal_non_normal_kaggle_dict.pkl', 'rb'))
for key, value in formal_xray_dict.items():
    for image_name, info in value['image_dict'].items():
        if 'AP' or 'PA' in info['type']:
            non_normal_list_kaggle.append((info['path'], key + '_' + image_name, 0))

print(len(normal_list))
print(len(non_normal_list_covid))
print (len(non_normal_list_kaggle))

random.shuffle(non_normal_list_kaggle)

non_normal_list = non_normal_list_covid + non_normal_list_kaggle[:3000]
random.shuffle(normal_list)
random.shuffle(non_normal_list)

np = len(normal_list)
nn = len(non_normal_list)

norm1 = int(0.5 * np)
norm2 = int(0.75 * np)

non_norm1 = int(0.5 * nn)
non_norm2 = int(0.75 * nn)

valid_norm_list_standard = normal_list[norm1:norm2]
test_norm_list_standard = normal_list[norm2:]

train_norm_list = []
norm_list = []
for case in normal_list[:norm1]:
    norm_list.append(case + ('original',))
    norm_list.append(case + ('fz_horizontal',))
    norm_list.append(case + ('fz_vertical',))
    norm_list.append(case + ('random_crop1',))
    norm_list.append(case + ('random_crop2',))
    norm_list.append(case + ('scale_0.5',))
    norm_list.append(case + ('scale_2',))
    norm_list.append(case + ('gaussian_0_1',))
    norm_list.append(case + ('gaussian_05_1',))
    norm_list.append(case + ('gaussian_50_1',))
train_norm_list = norm_list

valid_norm_list = []
norm_list = []
for case in normal_list[norm1:norm2]:
    norm_list.append(case + ('original',))
    norm_list.append(case + ('fz_horizontal',))
    norm_list.append(case + ('fz_vertical',))
    norm_list.append(case + ('random_crop1',))
    norm_list.append(case + ('random_crop2',))
    norm_list.append(case + ('scale_0.5',))
    norm_list.append(case + ('scale_2',))
    norm_list.append(case + ('gaussian_0_1',))
    norm_list.append(case + ('gaussian_05_1',))
    norm_list.append(case + ('gaussian_50_1',))
valid_norm_list = norm_list

test_norm_list = []
norm_list = []
for case in normal_list[norm2:]:
    norm_list.append(case + ('original',))
    norm_list.append(case + ('fz_horizontal',))
    norm_list.append(case + ('fz_vertical',))
    norm_list.append(case + ('random_crop1',))
    norm_list.append(case + ('random_crop2',))
    norm_list.append(case + ('scale_0.5',))
    norm_list.append(case + ('scale_2',))
    norm_list.append(case + ('gaussian_0_1',))
    norm_list.append(case + ('gaussian_05_1',))
    norm_list.append(case + ('gaussian_50_1',))
test_norm_list = norm_list

train_non_norm_list = non_normal_list[:non_norm1]
valid_non_norm_list = non_normal_list[non_norm1:non_norm2]
test_non_norm_list = non_normal_list[non_norm2:]

train_list = train_norm_list + train_non_norm_list
valid_list = valid_norm_list + valid_non_norm_list
test_list = test_norm_list + test_non_norm_list
valid_list_standard = valid_norm_list_standard + valid_non_norm_list
test_list_standard = test_norm_list_standard + test_non_norm_list

random.shuffle(train_list)

exp_data_id = 'standard_data_augmentation_nnn'
exp_data_dir = os.path.join('./data_preprocess', exp_data_id)
os.mkdir(exp_data_dir)


with open(os.path.join(exp_data_dir, 'data_statistic.csv'), 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ['N of Train', len(train_list), 'N of Norm', len(train_norm_list), 'N of Non Norm', len(train_non_norm_list)])
    csv_writer.writerow(
        ['N of Valid', len(valid_list), 'N of Norm', len(valid_norm_list), 'N of Non Norm', len(valid_non_norm_list)])
    csv_writer.writerow(
        ['N of Test', len(test_list), 'N of Norm', len(test_norm_list), 'N of Non Norm', len(test_non_norm_list)])
    csv_writer.writerow(
        ['N of Valid Standard', len(valid_list_standard), 'N of Norm', len(valid_norm_list_standard), 'N of Non Norm',
         len(valid_non_norm_list)])
    csv_writer.writerow(
        ['N of Test Standard', len(test_list_standard), 'N of Norm', len(test_norm_list_standard), 'N of Non Norm',
         len(test_non_norm_list)])

train_path = os.path.join(exp_data_dir, 'exp_train_list.pkl')
valid_path = os.path.join(exp_data_dir, 'exp_valid_list.pkl')
test_path = os.path.join(exp_data_dir, 'exp_test_list.pkl')
valid_path_standard = os.path.join(exp_data_dir, 'exp_valid_list_standard.pkl')
test_path_standard = os.path.join(exp_data_dir, 'exp_test_list_standard.pkl')

if os.path.exists(train_path):
    os.remove(train_path)

if os.path.exists(valid_path):
    os.remove(valid_path)

if os.path.exists(test_path):
    os.remove(test_path)

if os.path.exists(valid_path_standard):
    os.remove(valid_path_standard)

if os.path.exists(test_path_standard):
    os.remove(test_path_standard)

pickle.dump(train_list, open(train_path, 'wb'))
pickle.dump(valid_list, open(valid_path, 'wb'))
pickle.dump(test_list, open(test_path, 'wb'))
pickle.dump(valid_list_standard, open(valid_path_standard, 'wb'))
pickle.dump(test_list_standard, open(test_path_standard, 'wb'))

print('finished')