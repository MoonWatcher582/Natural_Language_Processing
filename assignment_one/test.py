from naive_bayes_model import NB
from distribution import Distribution

frequent_events_example = NB(Distribution('icecream_preference', {'chocolate' : 0.75, 'vanilla' : 0.25}),
                             {'chocolate' : [Distribution('wears_scarves', {'yes':0.7, 'no':0.3}),
                                      Distribution('has_bunny', {'yes':0.6, 'no':0.4})],
                              'vanilla' : [Distribution('wears_scarves', {'yes':0.4, 'no':0.6}),
                                      Distribution('has_bunny', {'yes':0.3, 'no':0.7})]})

frequent_events_data = [frequent_events_example.sample() for _ in range(1000)]

rare_events_example = NB(Distribution('icecream_preference', {'chocolate' : 0.75, 'vanilla' : 0.25}),
                             {'chocolate' : [Distribution('wears_scarves', {'yes':0.7, 'no':0.3}),
                                      Distribution('has_bunny', {'yes':0.6, 'no':0.4}),
                                      Distribution('grad_student', {'yes':0.4, 'no':0.6}),
                                      Distribution('favorite_language', {'python':0.6, 'java':0.4}),
                                      ],
                              'vanilla' : [Distribution('wears_scarves', {'yes':0.4, 'no':0.6}),
                                      Distribution('has_bunny', {'yes':0.3, 'no':0.7}),
                                      Distribution('grad_student', {'yes':0.6, 'no':0.4}),
                                      Distribution('favorite_language', {'python':0.8, 'java':0.2})]})

rare_events_data = [rare_events_example.sample() for _ in range(1000)]

#print frequent_events_example

#print "Frequent events: " 
#for event in frequent_events_data:
#	print event

#print "Rare events: " 
#for event in rare_events_data:
#	print event
