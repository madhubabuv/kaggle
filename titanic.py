#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import argparse

plt.style.use('classic')

def load_data(data_set_dir):

	data={}

	print data_set_dir+'train.csv'

	train=pd.read_csv(data_set_dir+'taitanic_train.csv')

	test=pd.read_csv(data_set_dir+'taitanic_test.csv')

	data['train']=train

	data['test']=test

	return data

def check_distrubution(train_data):

	ax = train_data["Age"].hist(color='teal', alpha=0.5)
	ax.set(xlabel='Age', ylabel='Count')
	plt.show()

	ax = train_data["Embarked"].hist(color='teal', alpha=0.5)
	ax.set(xlabel='Embarked', ylabel='Count')
	plt.show()

def handle_missing_values(data,mode,verbose=True):

	#missing value analysis values are missing in age,cabin,fare and embarked. lets solve one by one.

	if verbose:

		print "\nNull values percentage over the dataset \n",((data[mode].isnull().sum())/len(data[mode])*100)


		check_distrubution(data[mode])

	#since there the age distrubution is right skew we can replace the missing values with mean

	# 77 % values are missing in cabin so it would be better to drop the column

	# only two fare values missing we will replace them with mean

	# embarked values, most frequent variable S will be used in missing places

	age_median=data[mode]['Age'].median(skipna=True)
	fare_mean=data[mode]['Fare'].mean(skipna=True)

	if verbose:
		print "the age median is",age_median
		print "the fare mean is ",fare_mean
		print "the most frequent Embarked variable is S"
	data[mode]['Age'].fillna(age_median,inplace=True)

	data[mode]['Fare'].fillna(fare_mean,inplace=True)

	data[mode]['Embarked'].fillna('S',inplace=True)

	data[mode]=data[mode].drop('Cabin',axis=1)

	return data[mode]

def feature_engineering(data,verbose=False):

	#we should drop passenger name and ticket. they are not useful

	data=data.drop(['Name','Ticket'],axis=1)

	# make sibsp and parch into one categorical variable

	data['Family']=data['SibSp']+data['Parch']

	data['Alone']=np.where(data['Family']>0,0,1)

	# drop family and Sibsp,parch and family

	data=data.drop(['SibSp','Parch','Family'],axis=1)


	# convert passenger class,sex and embarked into one categorical variable

	data=pd.get_dummies(data,columns=['Pclass','Sex','Embarked'])

	#now drop sex_male of sex_female because it is redudent

	data=data.drop('Sex_male',axis=1)

	#drop passenger ID also because we dont need it. but for evaluation
	#  in the end we need so keep in separate variable



	return data

def Dtree(depth):

	from sklearn import tree

	dt=tree.DecisionTreeClassifier(max_depth=depth)

	return dt

def svm():

	from sklearn import svm

	clf=svm.SVC()

	return clf

def RFclf(depth):

	from sklearn.ensemble import RandomForestClassifier
	rf_clf=RandomForestClassifier(max_depth=depth, random_state=0)
	return rf_clf
def split_x_y(data):

	y=data['Survived']

	x=data.drop('Survived',axis=1)

	return x,y

def train_and_test(model,train,test):

	train_x,train_y=split_x_y(train)

	model.fit(train_x,train_y)

	pred_y=model.predict(test)

	test['Survived']=pred_y

	return test


def titanic(args):

	data=load_data(args.dataset_dir)

	#after loading data missing value analysis has to be done

	train_data=handle_missing_values(data,mode='train',verbose=args.verbose)

	test_data=handle_missing_values(data,mode='test',verbose=args.verbose)

	#After handling the missing values lets do feature engineering

	train_data=feature_engineering(train_data)

	#train_p_id=train_data['PassengerId']

	train_data=train_data.drop('PassengerId',axis=1)

	test_data=feature_engineering(test_data)

	test_p_id=test_data['PassengerId']

	test_data=test_data.drop('PassengerId',axis=1)

	#model=Dtree(3)

	#model=svm()

	model=RFclf(3)


	predicted_data=train_and_test(model,train_data,test_data)

	predicted_data['PassengerId']=test_p_id

	predicted_data[['PassengerId','Survived']].to_csv('titanic_survivors.csv',index=False)



if __name__=="__main__":

	parser = argparse.ArgumentParser(description="Titanic Survival Prediction")

	parser.add_argument("--dataset_dir", type=str, required=True, help="path to train and test csv")

	parser.add_argument("--verbose" ,type=bool,default=False,help="verbose the process")

	args = parser.parse_args()

	titanic(args)