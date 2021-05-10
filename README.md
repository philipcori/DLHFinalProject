# FLANNEL

## Data Prepare
### Data Collect
1. Download CCX data: from https://github.com/ieee8023/covid-chestxray-dataset, put them into original_data/covid-chestxray-dataset-master
2. Download KCX data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia, put them into original_data/chest_xray
### Data Preprocess
NOTE: for the following, 4-class data include classes (NORMAL, COVID-19, Viral Pneumonia, Bacterial Pneumonia), 3-class data include classes (COVID-19, Viral Pneumonia, Bacterial Pneumonia), 2-class data include (NORMAL, NON-NORMAL)
1. extract 4-class data from CCX: data_preprocess/get_covid_data_dict.py 
2. extract 4-class data from KCX: data_preprocess/get_kaggle_data_dict.py
3. extract 3-class data from CCX: data_preprocess/get_covid_3classes_data_dict.py
4. extract 3-class data from KCX: data_preprocess/get_kaggle_3classes_data_dict.py
5. extract 2-class data from CCX: data_preprocess/get_covid_normal_non_normal_data_dict.py
6. extract 2-class data from KCX: data_preprocess/get_kaggle_normal_non_normal_data_dict.py
7. reorganize 4-class CCX&KCX data to generate 5-folder cross-validation expdata: data_preprocess/extract_exp_data_crossentropy.py
8. reorganize 3-class CCX&KCX data to generate 5-folder cross-validation expdata: data_preprocess/extract_exp_data_crossentropy_3classes.py
9. reorganize 2-class CCX&KCX data to generate 5-folder cross-validation expdata: data_preprocess/extract_exp_data_crossentropy_nnn.py

## Model Training
### First-Stage Classifier
First Stage classifier is trained on distiguishing between two classes: normal vs. non-normal
```
# command to train the first-stage classifier
python -m explore_version_03.first_stage_classifier --arch resnet152 --epochs 200
```
### Second-Stage Classifier

FLANNEL 4 base-modeler learning [InceptionV3, Vgg19_bn, Resnet152, Densenet161]. Due to unavailability of resources, we have excluded resNeXt101_32x8d model. If resources are available, it is highly recommended to include this model among other baseline models to be trained.

#### Base-modeler Learning
Second-Stage Classifier is trained on distinguishing among 3 classes of non-normal patients: COVID-19, Viral Pneumonia, Bacterial Pneumonia.

* Train Model vgg19_bn using the following command ```python -m explore_version_03.second_stage_classifier --arch vgg19_bn --epochs 200```
* Train Model inception_v3 using the following command ```python -m explore_version_03.second_stage_classifier --arch inception_v3 --epochs 200 crop_size 299```
* Train Model resnet152 using the following command ```python -m explore_version_03.second_stage_classifier --arch resnet152 --epochs 200```
* Train Model densenet161 using the following command ```python -m explore_version_03.second_stage_classifier --arch densenet161 --epochs 200```

#### Predicting on individual baseline models

Predictions results on each individual baseline model will produce three CSV files: ```results_detail_ModelName_train_CrossValidationNumber.csv, results_detail_ModelName_valid_CrossValidationNumber.csv, results_detail_ModelName_test_CrossValidationNumber.csv``` These files are required in training the ensemble FLANNEL model. Generate Predictions according to the following steps

* Test Model vgg19_bn using the following command ```python -m explore_version_03.second_stage_classifier --arch vgg19_bn --epochs 200 --test```
* Test Model inception_v3 using the following command ```python -m explore_version_03.second_stage_classifier --arch inception_v3 --epochs 200 crop_size 299 --test```
* Test Model resnet152 using the following command ```python -m explore_version_03.second_stage_classifier --arch resnet152 --epochs 200 --test```
* Test Model densenet161 using the following command ```python -m explore_version_03.second_stage_classifier --arch densenet161 --epochs 200 --test```

### ensemble-model Learning
Train FLANNEL Model using the following command: ```python -m explore_version_03.ensemble_step2_second_stage_ensemble_learning --arch flannel --epochs 40```

### Predicting and combining the results




## Codes for
@misc{ 
      
      title={FLANNEL: An Improved Approach to Classify COVID-19 Chest X-ray Imaging Data}, 
      
      author={Samuel Youssef, Philip Cori, Katherine Ruiz, Erick Alanis}, 
      
      year={2021} 

}
