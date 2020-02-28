#load libraries

#statistics, dataframes, math
import numpy as np
import scipy as sp
import scipy.sparse
from scipy.io import arff
from scipy import stats
import sympy
import pandas as pd
import os
import datetime
from datetime import datetime
import statistics
from functools import reduce
import operator
import statsmodels.api as sm
import statsmodels.formula.api as smf

#visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams
%matplotlib inline
from pandas.plotting import register_matplotlib_converters
import seaborn as sns; sns.set()
sns.set_palette("pastel")
sns.set_style("whitegrid")
plt.style.use('seaborn-whitegrid')

#display preferences
import contextlib
import pandas.io.formats.format as pf
pd.options.display.float_format = '{:.3f}'.format

#machine learning
import sklearn
import sklearn.feature_extraction.text
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_importance
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree

#load files into pandas dataframes

#load the file of patients with no sepsis on arrival and no sepsis during their hospital stay.
no_hai_sepsis = pd.read_csv('/home/vanellope/Documents/Thinkful/Lesson_Code/final_capstone/no_hai_sepsis_charts4.csv',\
                         header=None, names=('subject_id', 'hadm_id', 'charttime', 'valuenum', 'valueuom', \
                                             'cat_group', 'cat_name', 'dob', 'admittime', 'dischtime', 'gender', \
                                             'hosp_expire_flag', 'class'))

#load the file of patients with no sepsis on arrival but with sepsis during their hospital stay.
hai_sepsis = pd.read_csv('/home/vanellope/Documents/Thinkful/Lesson_Code/final_capstone/hai_sepsis_charts3.csv',\
                         header=None, names=('subject_id', 'hadm_id', 'charttime', 'valuenum', 'valueuom', \
                                             'cat_group', 'cat_name', 'dob', 'admittime', 'dischtime', 'gender', \
                                             'hosp_expire_flag', 'class'))

#print the number of unique values of hadm_ids, which will serve as the 'n', population sample.
print('There are {} unique hospital stays during which a patient developed sepsis. \n'.\
      format(hai_sepsis['hadm_id'].nunique()))

print('There are {} unique hospital stays during which a patient did not develop sepsis. \n'.\
      format(no_hai_sepsis['hadm_id'].nunique()))

#combine the two dataframes into one dataframe.
sepsis = pd.concat([no_hai_sepsis, hai_sepsis])
print('Concatenation of hai_sepsis and no_hai_sepsis complete. \n')

#print the number of categories and patients in the merged dataframe.
print('There are {} unique categories in the sepsis dataframe. \n'.format(sepsis['cat_group'].nunique()))
print('There are {} unique patients in the sepsis dataframe. \n'.format(sepsis['subject_id'].nunique()))

#print information about the columns in the dataframe and take a look at the first few lines of the dataframe.
print(sepsis.info())
display(sepsis.head())

#dataframe cleaning with pandas

#create df with just hospital stay id (hadm_id) and class as part of an effort to reduce the number of hospital
#visits per class.
class_join = sepsis.drop(['subject_id', 'charttime', 'valuenum', 'valueuom', 'cat_group', 'cat_name', 'dob', \
                          'admittime', 'dischtime', 'gender', 'hosp_expire_flag'], axis=1)

#view the new dataframe head and information to make sure the code above worked.
display(class_join.head())
print(class_join.info()) #8,732,409 rows
print('There are {} unique hospital stay ids (hadm_id).\n'.format(class_join['hadm_id'].nunique()))
print('There are {} unique classes (sepsis and no sepsis)\n.'.format(class_join['class'].nunique()))

#develop a list of unique hadm_ids and corresponding class (sepsis or no sepsis) and then print the info.
class_join.drop_duplicates(keep='first', inplace=True)

print('Hadm_id information after duplicates removed:')
print(' \n')
print(class_join.info())
print(' \n')
print('Number of patients per class:{} \n'.format(class_join['class'].value_counts()))

#create class dataframes to make returning the first 1,213 of class 1 easier.
class_0 = class_join.loc[class_join['class'] != 1]
class_1 = class_join.loc[class_join['class'] != 0]

#keep first 1213 of each class
no_hai_sepsis_group = class_0[:1213]
hai_sepsis_group = class_1[:1213]
print('Description of no_hai_sepsis_group dataset:\n')
print(no_hai_sepsis_group.info())
display(hai_sepsis_group.head())
print('Description of hai_sepsis_group dataset:\n')
print(hai_sepsis_group.info())
display(hai_sepsis_group.head())

#combine the two dataframes into one dataframe.
hadm_class_list = pd.concat([no_hai_sepsis_group, hai_sepsis_group])
print('Unique hadm_ids with class \n')
print(hadm_class_list.info())
display(hadm_class_list.head())

#merge hadm_class_list with the full sepsis dataset via inner join to subset the sepsis dataset according to the 
#list of 1213 sepsis patients and 1213 non-sepsis patients. This step combines the reduced list of hospital stays
#with the chart data and patient demographics.
sepsis_cut = sepsis.merge(hadm_class_list, on='hadm_id')
print('Hadm_id and class dataframe has been merged with the sepsis dataframe to create the sepsis_cut dataframe.')

#get basic information about the dataframe and look at the top 5 and bottom 5 lines.
print('Sepsis_cut dataframe \n')
print(sepsis_cut.info()) #2,396,161 rows for 2426 people.
print(' \n')
print('Head: \n')
display(sepsis_cut.head())
print(' \n')
print('Tail: \n')
display(sepsis_cut.tail())

###handle null values

#check the percent null values per column as a first step in handling null values.
print((sepsis_cut.isnull().sum()*100)/sepsis_cut.isnull().count())

#drop null values. just a few in the numeric values and the value units of measure.
sepsis2 = sepsis_cut.dropna()

#a second class column was produced during the merge of hadm_class_list and the full data set. drop second class
#column and rename the remaining one.
pd.set_option('mode.chained_assignment', None)
sepsis2.drop(['class_y'], axis=1, inplace=True)
sepsis2.rename(columns={'class_x': 'class'}, inplace=True)

#print head of the clean dataframe to ensure second class column was removed.
display(sepsis2.head())

### Convert data type of select columns

#convert datetime columns from data type object to data type datetime.
pd.set_option('mode.chained_assignment', None)
sepsis2['charttime'] = pd.to_datetime(sepsis2.charttime)
sepsis2['dob'] = pd.to_datetime(sepsis2.dob)
sepsis2['admittime'] = pd.to_datetime(sepsis2.admittime)
sepsis2['dischtime'] = pd.to_datetime(sepsis2.dischtime)

### Create new features from existing columns

#### Length of Stay

#calculate length of stay
sepsis2['los'] = sepsis2[['dischtime', 'admittime']].apply(lambda x: (x['dischtime'] - x['admittime']), axis=1)

#view the high and low values of 'los' (length of stay)
display(sepsis2.sort_values(by='los', ascending=False))

display(sepsis2.sort_values(by='los', ascending=True))

**The longest length of stay is 191 days and the shortest is -1 days, the latter of which is an error.**

# Filter out all rows for which the length of stay is <=0 because, somehow, there were some rows of data for which 
#the charttime was earlier than the admittime.
sepsis2_filtered = sepsis2[sepsis2['los'] >= '0 days +00:00:00']

#check to make sure negative length of stay (los) values were removed from the dataset.
display(sepsis2_filtered.sort_values(by='los', ascending=True))

#create column 'los_days' by extracting day from los.
sepsis2_filtered['los_days'] = sepsis2_filtered.los.dt.days

#### Age

#create column admityear by extracting year from admittime.
sepsis2_filtered['admityear'] = sepsis2_filtered.admittime.dt.year

#create column yob by extracting year from dob.
sepsis2_filtered['yob'] = sepsis2_filtered.dob.dt.year

#create column age by subtracting yob from admityear.
sepsis2_filtered['age'] = \
    sepsis2_filtered[['admityear', 'yob']].apply(lambda x: (x['admityear'] - x['yob']), axis=1)

#adjusted age
#from MIMIC-III documentation: 'DOB is the patient’s date of birth. If the patient is older than 89, their age is 
#set to 300 at their first admission.'
sepsis2_filtered['adj_age'] = sepsis2_filtered['age']

sepsis2_filtered.reset_index(drop=True, inplace=True)

sepsis2_filtered.loc[sepsis2_filtered['age'] >= 300, 'adj_age'] = sepsis2_filtered['age'] - 210

#### Chart Event Day

#calculate days_after_admit by subtracting the admittime from the charttime.
sepsis2_filtered['days_after_admit'] = sepsis2_filtered[['charttime', 'admittime']].apply(lambda x: (x['charttime'] - x['admittime']), axis=1)

sepsis2_filtered.sort_values(by='days_after_admit', ascending=False)

#create column day by extracting days from the days_after_admit timedelta. day is going to be used for grouping for
#aggregate functions.
sepsis2_filtered['day'] = sepsis2_filtered.days_after_admit.dt.days

#Filter out all rows for which the day of a chart event is <=0 because, somehow, there were some rows of data for 
#which the charttime was earlier than the admittime.
sepsis3 = sepsis2_filtered[sepsis2_filtered['day'] >= 0]

# Filter out all rows for which the length of stay is >=60 because I need to limit the amount of features in the 
#data set. Then check the dataframe info.
sepsis4 = sepsis3[sepsis3['day'] <= 60]
sepsis4.info()

#### Vital Sign Day

#convert hadm_id and day to strings and then combine them into a new feature called vitalsign_day. This is part of 
#my effort to 'flatten' the data set to return dataframe with one row for each hadm_id with all the data from that 
#hospital stay in one row.
sepsis4['hadm_id'] = sepsis4['hadm_id'].astype('str')
sepsis4['day'] = sepsis4['day'].astype('str')
sepsis4['vitalsign_day'] = sepsis4[['cat_name', 'day']].apply(lambda x: (x['cat_name'] + '_' + 'day' + x['day']), axis=1)

#check to make sure the new variables are in the dataframe.
display(sepsis4.head())

#### Aggregating Vital Sign Measurements by Hospital Stay and Vital-Sign-Day

#drop columns used to create new features or to improve readability of the dataframe but not necessary for modeling.
#drop some additional columns for the pivot_table function. i will join them back to the resulting table.
sepsis_stats = sepsis4.drop(['valueuom', 'cat_name', 'charttime', 'dob', 'admittime', 'dischtime', 'subject_id', \
                        'gender', 'hosp_expire_flag', 'class', 'los', 'los_days', 'admityear', 'yob', \
                        'age', 'adj_age', 'days_after_admit', 'cat_group', 'day'], axis = 1)

sepsis_stats.info()

#reorder the columns
sepsis_stats = sepsis_stats[['hadm_id', 'vitalsign_day', 'valuenum']]

#create dataframe of columns that will be rejoined to the concatenated pivot table below.
sepsis_for_join = sepsis4.drop(['valuenum', 'valueuom','cat_name', 'charttime', 'dob', 'admittime', 'dischtime', \
                                'subject_id','vitalsign_day', 'los', 'admityear', 'yob', 'age', 'days_after_admit',\
                                'cat_group', 'day'], axis = 1)

sepsis_for_join['hadm_id'] = sepsis_for_join['hadm_id'].astype('str')
sepsis_for_join2 = sepsis_for_join.drop_duplicates(keep='first', subset='hadm_id', inplace=False)

sepsis_for_join2.head()

#create 3 pivot tables: mean, max, min. replace 'valuenum' with the type of statistic calculated (mean, max, min).

#mean
sepsis_mean = pd.pivot_table(sepsis_stats, index=['hadm_id', 'vitalsign_day'], aggfunc='mean')
sepsis_mean.reset_index(drop=False, inplace=True)
sepsis_mean.rename(columns={'valuenum': 'mean'}, inplace=True)

#max
sepsis_max = pd.pivot_table(sepsis_stats, index=['hadm_id', 'vitalsign_day'], aggfunc='max')
sepsis_max.reset_index(drop=False,inplace=True)
sepsis_max.rename(columns={'valuenum': 'max'}, inplace=True)

#min
sepsis_min = pd.pivot_table(sepsis_stats, index=['hadm_id', 'vitalsign_day'], aggfunc='min')
sepsis_min.reset_index(drop=False,inplace=True)
sepsis_min.rename(columns={'valuenum': 'min'}, inplace=True)

# Using DataFrame.insert() to add a column.
sepsis_mean['statistic'] = sepsis_mean.apply((lambda x: 'mean'), axis=1)
sepsis_mean2 = sepsis_mean.rename(columns={'mean':'value'})

sepsis_max['statistic'] = sepsis_max.apply((lambda x: 'max'), axis=1)
sepsis_max2 = sepsis_max.rename(columns={'max':'value'})

sepsis_min['statistic'] = sepsis_min.apply((lambda x: 'min'), axis=1)
sepsis_min2 = sepsis_min.rename(columns={'min':'value'})

#display tables before merging to make sure the column was added and renamed correctly in each table.
display(sepsis_mean2.head())
display(sepsis_max2.head())
display(sepsis_min2.head())

#combine the three dataframes into one dataframe.
sepsis_stats = pd.concat([sepsis_mean2, sepsis_max2, sepsis_min2])
print(sepsis_stats.info())
display(sepsis_stats.head())

#combine statistic column with vitalsign_day column.
sepsis_stats['stat_vitalsign_day'] = sepsis_stats[['vitalsign_day', 'statistic']].\
    apply(lambda x: (x['statistic'] + x['vitalsign_day']), axis=1)

sepsis_stats.drop(['vitalsign_day', 'statistic'], axis=1, inplace=True)

#reorder the columns
sepsis_stats = sepsis_stats[['hadm_id', 'stat_vitalsign_day', 'value']]

display(sepsis_stats.head())

sepsis_stats.info() #124,281 rows of data

sepsis_stats.nunique()

#saved the dataframe to disk to be able to pick up here if necessary.
sepsis_stats.to_csv('sepsis_stats.csv')

sepsis_stats2 = pd.read_csv('/home/vanellope/Documents/Thinkful/Lesson_Code/final_capstone/sepsis_stats.csv')

sepsis_stats2.head()

#remove white spaces from column names and remove extra column that was added after saving the dataframe to csv.
sepsis_stats2 = sepsis_stats2.rename(columns=lambda x: x.strip())

sepsis_stats2.drop(['Unnamed: 0'], axis=1, inplace=True)

sepsis_stats2.head()

#pivot the dataframe to produce a new dataframe with only one row per hadm_id.
sepsis_stats3 = sepsis_stats2.pivot(index='hadm_id', columns='stat_vitalsign_day', values=['value'])

print(sepsis_stats3.info())
display(sepsis_stats3.head())

#the dataframe pivot above created a multiindex in the new dataframe.
#remove the multiindex
sepsis_stats3.reset_index(inplace=True)

display(sepsis_stats3.head())

#the multiindex still exists. here I used a temporary dataframe to fix the sepsis_stats3 dataframe so it has a 
#single index.
#hadm_id ends up on the far right of the table.
my_new_df = sepsis_stats3['value']

my_new_df['hadm_id'] = sepsis_stats3['hadm_id'].astype('int64')
print(my_new_df.info())
display(my_new_df.head())

#join the categorical patient data to the aggregation data.
sepsis_for_join2['hadm_id'] = sepsis_for_join2['hadm_id'].astype('int64')
sepsis_analysis = pd.merge(my_new_df, sepsis_for_join2, on='hadm_id', how='inner')
display(sepsis_analysis.head())

#change gender categories to numbers for further analysis
sepsis_analysis['gender'] = sepsis_analysis['gender'].apply(lambda x: 1 if x == 'F' else 0)

#save sepsis_analysis dataframe as a csv file. this is the dataframe that will be used for modeling below.
sepsis_analysis.to_csv('sepsis_analysis.csv')

### Distributions and Counts

sns.set(font_scale=1.5, palette='viridis_r', rc={'figure.figsize':(4,6)})
g4 = sns.countplot(
    x="class", data=sepsis_for_join2).set_title(
    "Number of Hospital Visits With and Without Development of Sepsis", y=1.08)
plt.savefig('hospital_visits.png')

#plot age distribution according to class.
sns.set(font_scale=1.5, palette='viridis_r') 
g1 = sns.FacetGrid(sepsis_for_join2, col="class", hue="class", size=6)
g1.map(plt.hist, "adj_age")
g1.set_axis_labels(x_var="Age", y_var="Number of People")
g1.fig.subplots_adjust(top=0.825)
g1.fig.suptitle("Age Distributions", fontsize=22)
axes = g1.axes.flatten()
axes[0].set_title("No Sepsis")
axes[1].set_title("Sepsis")
plt.savefig('age_distributions.png')

#plot length of stay according to class.
g2 = sns.FacetGrid(sepsis_for_join2, col="class", hue="class", size=6)
g2.map(plt.hist, "los_days")
g2.set_axis_labels(x_var="Length of Stay (days)", y_var="Number of People")
g2.fig.subplots_adjust(top=0.825)
g2.fig.suptitle("Length-of-Stay Distributions", fontsize=22)
axes = g2.axes.flatten()
axes[0].set_title("No Sepsis")
axes[1].set_title("Sepsis")
plt.savefig('los_distributions.png')

sns.set(font_scale=1.5, palette='viridis_r', rc={'figure.figsize':(6.5,6)}) 

g3 = sns.countplot(
    x="class", hue="hosp_expire_flag", data=sepsis_for_join2).set_title(
    "Mortality of Patients With and Without Sepsis", fontsize=22, y=1.08)
plt.legend(loc='best', labels=['Lived', 'Died'])
plt.savefig('mortality.png')

sepsis_for_join2['hadm_id'].nunique()

# Classification modeling

## Load and Prepare Data

#load data
sepsis_analysis = pd.read_csv('/home/vanellope/Documents/Thinkful/Lesson_Code/final_capstone/sepsis_analysis.csv')

#replacing nan values in dataframe with -999.0
sepsis_analysis.fillna(-999.0, inplace = True)

#replacing white spaces in the column names with underscores (_).
sepsis_analysis = sepsis_analysis.rename(columns = lambda x: x.replace(' ', '_'))

#separate data from target. also remove columns that were added unintentionally and those that aren't really
#features, but rather dataset identifiers or outcomes not relevant to the modeling (mortality/hosp_expire_flag).
X = sepsis_analysis.drop(columns=['Unnamed:_0', 'hadm_id', 'class', 'hosp_expire_flag'], \
                         axis=1) 
y = sepsis_analysis['class']

#split data into train and test sets
#we also specify a seed for the random number generator so that we always get the same split of data each time this
#is executed.
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Random Forest

## Train the Random Forest Model

#fit model on training data
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

## Make Predictions with Random Forest Model

#make predictions with training data
rf_train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
print("Accuracy: %.2f%%" % (rf_train_accuracy * 100.0))

#make predictions for test data
rf_predictions = rf_model.predict(X_test)

#evaluate predictions
rf_test_accuracy = accuracy_score(y_test, rf_predictions)
print("Accuracy: %.2f%%" % (rf_test_accuracy * 100.0))

# Tune Hyperparameters

## Number of Trees and Tree Depth

#use grid search to evaluate 4 options for number of trees and 4 options for depth of each tree.
rf_model = RandomForestClassifier()
n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]
print(max_depth)
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
rf_grid_search = GridSearchCV(rf_model, param_grid, n_jobs=4, cv=kfold, verbose=1)
rf_grid_result = rf_grid_search.fit(X_train, y_train)

#summarize results of grid search
print("Best: %f using %s" % (rf_grid_result.best_score_, rf_grid_result.best_params_))
means_rf = rf_grid_result.cv_results_[ 'mean_test_score' ]
stds_rf = rf_grid_result.cv_results_[ 'std_test_score' ]
params_rf = rf_grid_result.cv_results_[ 'params' ]
for mean, stdev, param in zip(means_rf, stds_rf, params_rf):
    print("%f (%f) with: %r" % (mean, stdev, param))

#plot results
scores_rf = np.array(means_rf).reshape(len(max_depth), len(n_estimators))
for i, value in enumerate(max_depth):
    plt.rcParams["figure.figsize"] = (10,10)
    plt.plot(n_estimators, scores_rf[i], label= ' depth: ' + str(value))
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0))
    plt.xlabel( ' n_estimators ' )
    plt.ylabel( ' Log Loss ' )
    plt.savefig('n_estimators_vs_max_depth.png')

**Grid search identified 8 as the ideal depth of each tree and 100 as the ideal number of trees. However, there is very little difference between a depth of 6 and a depth of 8, as seen in the plot above. Below I instantiated the final random forest model using these hyperparameters.**

#fit model on training data
rf_model_final = RandomForestClassifier(max_depth=8, n_estimators=100, verbose=1)
rf_model_final.fit(X_train, y_train)

#make predictions on validation dataset
rf_final_predictions = rf_model_final.predict(X_test)
print("Accuracy: %.2f%%" % (accuracy_score(y_test, rf_final_predictions) * 100.0))
print(confusion_matrix(y_test, rf_final_predictions))
print(classification_report(y_test, rf_final_predictions))

**There was a drastic difference between accuracy with the training data (98.16%) and accuracy with the testing data (66.41%), suggesting overfitting of the model. The accuracy of the final model was 68.47%. Also, the recall for patients who developed sepsis was quite low at only 0.59. It is most important to diagnose all patients who will develop sepsis so administration of antibiotics and other measures can be taken as soon as possible.**

# XGBoost

## Train the XGBoost Model

#fit model on training data
xgb_model = XGBClassifier(missing=-999.0)
xgb_model.fit(X_train, y_train)

## Make Predictions with XGBoost Model

#make predictions with training data
xgb_train_accuracy = accuracy_score(y_train, xgb_model.predict(X_train))
print("Accuracy: %.2f%%" % (xgb_train_accuracy * 100.0))

#make predictions for test data
xgb_predictions = xgb_model.predict(X_test)

#evaluate predictions
xgb_test_accuracy = accuracy_score(y_test, xgb_predictions)
print("Accuracy: %.2f%%" % (xgb_test_accuracy * 100.0))

# Tune Hyperparameters

## Number of Trees and Tree Depth

#use grid search to evaluate 4 options for number of trees and 4 options for depth of each tree.
xgb_model1 = XGBClassifier(missing=-999.0)
n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]
print(max_depth)
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
xgb_grid_search1 = GridSearchCV(xgb_model1, param_grid, scoring="neg_log_loss", n_jobs=4, cv=kfold, verbose=1)
xgb_grid_result1 = xgb_grid_search1.fit(X_train, y_train)

#summarize grid search results
print("Best: %f using %s" % (xgb_grid_result1.best_score_, xgb_grid_result1.best_params_))
means_xgb1 = xgb_grid_result1.cv_results_[ 'mean_test_score' ]
stds_xgb1 = xgb_grid_result1.cv_results_[ 'std_test_score' ]
params_xgb1 = xgb_grid_result1.cv_results_[ 'params' ]
for mean, stdev, param in zip(means_xgb1, stds_xgb1, params_xgb1):
    print("%f (%f) with: %r" % (mean, stdev, param))

#plot results
scores_xgb1 = np.array(means_xgb1).reshape(len(max_depth), len(n_estimators))
for i, value in enumerate(max_depth):
    plt.rcParams["figure.figsize"] = (10,10)
    plt.plot(n_estimators, scores_xgb1[i], label= ' depth: ' + str(value))
    plt.legend()
    plt.xlabel( ' n_estimators ' )
    plt.ylabel( ' Log Loss ' )
    plt.savefig('n_estimators_vs_max_depth.png')

**Grid search identified 2 as the ideal depth of each tree and 100 as the ideal number of trees.**

## Learning Rate and Number of Trees

#use grid search to evaluate 4 options for number of trees and 4 options for learning rate.
xgb_model2 = XGBClassifier(missing=-999.0)
n_estimators = [100, 200, 300, 400]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
xgb_grid_search2 = GridSearchCV(xgb_model2, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
xgb_grid_result2 = xgb_grid_search2.fit(X_train, y_train)

#summarize results
print("Best: %f using %s" % (xgb_grid_result2.best_score_, xgb_grid_result2.best_params_))
means_xgb2 = xgb_grid_result2.cv_results_[ 'mean_test_score' ]
stds_xgb2 = xgb_grid_result2.cv_results_[ 'std_test_score' ]
params_xgb2 = xgb_grid_result2.cv_results_[ 'params' ]
for mean, stdev, param in zip(means_xgb2, stds_xgb2, params_xgb2):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
#plot results
scores_xgb2 = np.array(means_xgb2).reshape(len(learning_rate), len(n_estimators))
for i, value in enumerate(learning_rate):
    plt.plot(n_estimators, scores_xgb2[i], label= ' learning_rate: ' + str(value))
    plt.legend()
    plt.xlabel( ' n_estimators ' )
    plt.ylabel( ' Log Loss ' )
    plt.savefig('n_estimators_vs_learning_rate.png')

**The two grid searches returned two different ideal number of trees. The third grid search below includes the two previously chosen number of trees, 100 and 400, and keeps the tree depth and learning rate at the chosen values, 2 and 0.01, respectively.**

#grid search
xgb_model3 = XGBClassifier(missing=-999.0)
n_estimators = [100, 400]
max_depth = [2]
learning_rate = [0.01]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
xgb_grid_search3 = GridSearchCV(xgb_model3, param_grid, scoring="neg_log_loss", n_jobs=4, cv=kfold)
xgb_grid_result3 = xgb_grid_search3.fit(X_train, y_train)

#summarize results
print("Best: %f using %s" % (xgb_grid_result3.best_score_, xgb_grid_result3.best_params_))
means_xgb3 = xgb_grid_result3.cv_results_[ 'mean_test_score' ]
stds_xgb3 = xgb_grid_result3.cv_results_[ 'std_test_score' ]
params_xgb3 = xgb_grid_result3.cv_results_[ 'params' ]
for mean, stdev, param in zip(means_xgb3, stds_xgb3, params_xgb3):
    print("%f (%f) with: %r" % (mean, stdev, param))

#fit final model on training data with hyperparameters from tuning.
xgb_model_final = XGBClassifier(max_depth=2, learning_rate=0.01, n_estimators=400, objective='binary:logistic', \
                            missing=-999.0)
xgb_model_final.fit(X_train, y_train)

#make predictions with final model.
xgb_final_predictions = xgb_model_final.predict(X_test)
print("Accuracy: %.2f%%" % (accuracy_score(y_test, xgb_final_predictions) * 100.0))
print(confusion_matrix(y_test, xgb_final_predictions))
print(classification_report(y_test, xgb_final_predictions))

**There was less of a difference between accuracy with the training data (79.76%) and accuracy with the testing data (69.50%), suggesting the XGBoost model did not overfit. Tuning the hyperparameters did not change the accuracy substantially, with 69.50%% before tuning and 68.47% after tuning. The recall for patients who developed sepsis was somewhat higher for the XGBoost model at 0.64 compared to 0.59 for the Random Forest model.**

#plot the features according to importance in descending order
plt.rcParams["figure.figsize"] = (10,10)
plt.rc('xtick', labelsize=20, color='g') 
plt.rc('ytick', labelsize=20)
plot_importance(xgb_model_final, max_num_features=10)
plt.savefig('feature_importances', dpi=300)