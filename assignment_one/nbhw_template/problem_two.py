from naive_bayes_model import NB
from distribution import Distribution
from observations import *
from test import frequent_events_data
import random, collections

class id_node:
	def __init__(self, identifier):
		self.identifier = identifier
		self.values = dict() #dict of {value:count}
		self.children = dict() #dict of {value:{feature:{subvalue:count}}}
		self.total_count = 0

def NB_from_data(prediction, data):
	#given frequent_events_data, create the NB object that goes into, say frequent_events_example
	head = count_data(prediction, data)
	head = probabilitize_data(head)
	
	prior = Distribution(head.identifier, head.values)

	likelihood = dict()
	for prior_value in head.children:
		likelihood[prior_value] = list()
		for feat in head.children[prior_value]:
			variable = feat
			table = head.children[prior_value][feat]
			likelihood[prior_value].append(Distribution(variable, table))

	bayes = NB(prior, likelihood) 		
	
	return bayes

def probabilitize_data(node):
	sum = 0
	for key, value in node.values.items():
		sum += value
	for key in node.values:
		node.values[key] = node.values[key] / (sum * 1.0)

	for value, feat_dict in node.children.items():
		for feat, subval_dict in node.children[value].items():
			sum = 0
			for subvalue, count in node.children[value][feat].items():
				sum += count
			for subvalue in node.children[value][feat]:
				node.children[value][feat][subvalue] = node.children[value][feat][subvalue] / (sum * 1.0)

	#print node.identifier
	#print node.values
	#print node.children
	
	return node

def count_data(prediction, data):
	head = init_data(prediction, data)
	for datum in data:
		pred_key = head.identifier
		pred_value = datum[prediction]
		for key, value in datum.items():
			if key == prediction:
				head.values[value] += 1
			else:
				head.children[pred_value][key][value] += 1
	
	#print head.identifier
	#print head.values
	#print head.children
	
	return head			

def init_data(prediction, data):
	"""creates a data structure to parse and count the data"""
	feats = list(get_features(data))
	head = id_node(prediction)
	for value in get_values(prediction, data):
		head.values[value] = 0
		head.children[value] = dict()
		for i in range(len(feats)):
			if feats[i] == prediction:
				continue
			head.children[value][feats[i]] = dict()
			for subvalue in get_values(feats[i], data):
				head.children[value][feats[i]][subvalue] = 0
	
	#print head.identifier
	#print head.values
	#print head.children
	
	return head

print NB_from_data('icecream_preference', frequent_events_data)
