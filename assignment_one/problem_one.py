from observations import *
import random, collections
import sys

def classify_from_data(target, data, guess):
	def classify(event):
		#find the items in data that match the event
		matches = get_matches(event, data)
		#make a table of the different outcomes of target and how often they occur
		different_outcomes = {}
		for e in matches:
			for feat, value in e:
				if value in different_outcomes and feat == target:
					different_outcomes[value] += 1
				elif feat == target:
					different_outcomes[value] = 1
		#find the most likely possibility in the table, if any, and return it
		max_count = 0
		for possibility, count in different_outcomes:
			if count > max_count:
				max_count = count
				max_possibility = possibility
		if max_possibility != None:
			return max_possibility
		return guess
	return classify
