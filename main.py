import pandas
from math import log
import numpy as np
import pandas as pd
from scipy.linalg.decomp_schur import eps
from sklearn.model_selection import KFold


def find_total_entropy(dataframe):
	# take target values = Class[Positive, Negative]
	Class = dataframe.keys()[-1]
	target_values = dataframe[Class].unique()
	total_entropy = 0

	# for each target value like Positive, Negative calculate entropy, add to total
	for value in target_values:
		proportion = dataframe[Class].value_counts()[value] / len(dataframe[Class])
		total_entropy += -proportion * np.log2(proportion)

	return total_entropy


def find_entropy_attribute(dataframe, column):
	# target value
	Class = dataframe.keys()[-1]
	# get target (Class) values -> Positive, Negative
	target_variables = dataframe[Class].unique()
	# get values for current column
	column_variables = dataframe[column].unique()

	column_total_entropy = 0
	for variable in column_variables:
		entropy = 0
		for target_variable in target_variables:
			# find number of columns where target variable = variable and current column variable = variable
			# ex. Class = Positive and Age = Above Average for the current training set
			numerator = len(dataframe[column][dataframe[column] == variable][dataframe[Class] == target_variable])
			# find number of columns where target variable = variable
			denominator = len(dataframe[column][dataframe[column] == variable])
			# find the proportion
			proportion = numerator / (denominator + eps)
			# get entropy for current values
			entropy += -proportion * log(proportion + eps)
		fraction2 = denominator / len(dataframe)
		# add to total entropy
		column_total_entropy += -fraction2 * entropy
	return abs(column_total_entropy)


def find_max_info_gain(df):
	information_gains = []
	for key in df.keys()[:-1]:  # do not include target variable
		# Gain(S,A) = Entropy(S) - sum(|S column|/|S|) Entropy(S column)
		information_gains.append(find_total_entropy(df) - find_entropy_attribute(df, key))
	# return max info gain = most effective attribute in classifying training data
	return df.keys()[:-1][np.argmax(information_gains)]


# this function gets sub dataframe including columns where node = given value
def get_sub_dataframe(df, node, value):
	return df[df[node] == value].reset_index(drop=True)


def build_tree(df, tree=None):
	# take target value = Class[Positive, Negative]
	Class = df.keys()[-1]
	# in order to decide next node, find attribute with max information gain
	node = find_max_info_gain(df)
	# get distinct values for that attribute like Yes, No, Positive, Negative
	distinct_values = df[node].unique()

	# create empty dictionary for tree representation
	if tree is None:
		tree = {}
		tree[node] = {}

	for value in distinct_values:
		sub_data_frame = get_sub_dataframe(df, node, value)
		# BURADAN EMİN DEĞİLİM!! DATAFRAME BOŞ GELİNCE NE YAPILACAK?
		if sub_data_frame.empty:
			continue
		clValue, counts = np.unique(sub_data_frame[Class], return_counts=True)
		if len(counts) == 1:  # check for purity
			tree[node][value] = clValue[0]
			df = data_train_2[data_train_2[node] == value].reset_index(drop=True)
		else:
			tree[node][value] = build_tree(sub_data_frame)
	return tree


def visualize(root, indent=0):
	if type(root) == dict:
		for k, v in root.items():
			print(" " * indent + f"{k}:")
			visualize(v, indent + 2)
	else:
		print(" " * indent + repr(root))


def predict(tree, test_sample):
	if not isinstance(tree, dict):  # if it is leaf node
		return tree  # return the value
	else:
		root_node = next(iter(tree))  # getting first key/feature name of the dictionary
		feature_value = test_sample[root_node]  # value of the feature
		if feature_value in tree[root_node]:  # checking the feature value in current tree node
			return predict(tree[root_node][feature_value], test_sample)  # goto next feature
		else:
			return None


def evaluate(tree, test_data, label):
	correct_prediction = 0
	incorrect_prediction = 0
	for index, row in test_data.iterrows():  # for each row in the dataset
		#print(test_data.iloc[index])
		result = predict(tree, test_data.iloc[index])  # predict the row
		if result == test_data[label].iloc[index]:  # predicted value and expected value is same or not
			correct_prediction += 1  # increase correct count
		else:
			incorrect_prediction += 1  # increase incorrect count
	accuracy = correct_prediction / (correct_prediction + incorrect_prediction)  # calculating accuracy
	return accuracy


def k_fold_cross_validation(data):
	global data_train_2
	# define sklearn KFold
	kf = KFold(n_splits=5, shuffle=True)
	# split data frame to 5
	kf.split(data)
	# loop over 5 different test & training data, call function
	for train_index, test_index in kf.split(data):
		data_train, data_test = data.iloc[train_index].reset_index(drop=True), data.iloc[test_index].reset_index(drop=True)
		print("############################################################")
		print()
		data_train_2 = data_train
		data_train.to_csv("result.csv")
		# build tree
		tree = build_tree(data_train)
		#visualize(tree)
		accuracy = evaluate(tree, data_test, "Class")
		print(accuracy)

		print()
		print("############################################################")

	# print("----------------------------------------")
	# print(data_test)
	# print(data_train)
	# print("----------------------------------------")


def main():
	data = pd.read_csv('diabetes_data_upload.csv')
	# discretization for continuous variable Age, split Age values to 5 different ranges -> make it discrete
	data['age_discrete'] = pd.cut(data['Age'], 5, labels=['Low', 'Below_average', 'Average', 'Above_Average', 'High'])
	# write discretized age values to Age column
	data['Age'] = data['age_discrete']
	# drop unnecessary column
	data = data.drop(columns="age_discrete")
	k_fold_cross_validation(data)


data_train_2 = pandas.DataFrame
main()
