import pandas
from math import log
import numpy as np
import pandas as pd
from scipy.linalg.decomp_schur import eps
from sklearn.model_selection import KFold


def find_total_entropy(df):
	Class = df.keys()[-1]  # To make the code generic, changing target variable class name
	entropy = 0
	values = df[Class].unique()
	for value in values:
		fraction = df[Class].value_counts()[value] / len(df[Class])
		entropy += -fraction * np.log2(fraction)
	return entropy


def find_entropy_attribute(df, attribute):
	Class = df.keys()[-1]  # To make the code generic, changing target variable class name
	target_variables = df[Class].unique()  # This gives all 'Yes' and 'No'
	variables = df[
		attribute].unique()  # This gives different features in that attribute (like 'Hot','Cold' in Temperature)
	entropy2 = 0
	for variable in variables:
		entropy = 0
		for target_variable in target_variables:
			num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
			den = len(df[attribute][df[attribute] == variable])
			fraction = num / (den + eps)
			entropy += -fraction * log(fraction + eps)
		fraction2 = den / len(df)
		entropy2 += -fraction2 * entropy
	return abs(entropy2)


def find_max_info_gain(df):
	Entropy_att = []
	IG = []
	for key in df.keys()[:-1]:
		Entropy_att.append(find_entropy_attribute(df, key))
		IG.append(find_total_entropy(df) - find_entropy_attribute(df, key))
	return df.keys()[:-1][np.argmax(IG)]


def get_sub_dataframe(df, node, value):
	return df[df[node] == value].reset_index(drop=True)


def kfoldize(data):
	global data_train
	kf = KFold(n_splits=5, shuffle=False)
	kf.split(data)
	for train_index, test_index in kf.split(data):
		data_train, data_test = data.iloc[train_index], data.iloc[test_index]
		print("############################################################")
		print()
		data_train = data_train
		tree = build_tree(data_train)
		visualize(tree)
		print()
		print("############################################################")

		# print("----------------------------------------")
		# print(data_test)
		# print(data_train)
		# print("----------------------------------------")


def build_tree(df, tree=None):
	Class = df.keys()[-1]
	node = find_max_info_gain(df)
	attValue = df[node].unique()

	if tree is None:
		tree = {}
		tree[node] = {}

	for value in attValue:
		subtable = get_sub_dataframe(df, node, value)
		clValue, counts = np.unique(subtable[Class], return_counts=True)
		if len(counts) == 1: #check for purity
			tree[node][value] = clValue[0]
			df = data_train[data_train[node] == value].reset_index(drop=True)
		else:
			tree[node][value] = build_tree(subtable)
	return tree


def visualize(root, indent=0):
	if type(root) == dict:
		for k, v in root.items():
			print(" " * indent + f"{k}:")
			visualize(v, indent + 2)
	else:
		print(" " * indent + repr(root))


def main():
	global data
	data['age_discrete'] = pd.cut(data['Age'], 5, labels=['Low', 'Below_average', 'Average', 'Above_Average', 'High'])
	data['Age'] = data['age_discrete']
	data = data.drop(columns="age_discrete")
	data = data.iloc[:, 1:]
	data.to_csv("result.csv")
	kfoldize(data)


data_train = pandas.DataFrame
data = pd.read_csv('diabetes_data_upload.csv')
main()
