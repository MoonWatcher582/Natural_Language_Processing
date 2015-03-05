import random, collections

# events are of the form:
# [ event1, event2, ... ]
# an event is of the form:
# {feature1:value1, feature2:value2, ...}

def get_features(events): 
    """
    get_features(events): 
    iterate through the events, which are dictionaries, 
    and take the set union of all of their feature values
    using a set forces uniqueness of feature names
    """
    features = set() 
    for event in events: 
        features.update(event.keys()) #event.keys == list of features of event
    return features
        
def get_values(feature, events):
    """
    get_values(feature, events):
    get all possible values for given feature. 
    using a set forces uniqueness of feature values
    """
    values = set()
    for event in events:
        if feature in event:
            values.add(event[feature])
    return values
            
def count_f(feature, events): 
    """
    count_f(feature, events):
    count how many of the passed events 
    have the specified feature 
    """
    return sum(1 for event in events if feature in event)

def count_fv(feature, value, events): 
    """
    count_fv(feature, value, events): 
    count how many of the passed events
    have the specified value for the specified feature
    """
    return sum(1 for event in events if feature in event and event[feature] == value)

def split_f(feature, events): 
    """
    split_f(feature, events) :
    divide the passed events into groups based on the values
    they have for the passed feature.
    returns a dictionary of the form {featurevalue:[list of events]} 
    with a key featurevalue for each value for the specified feature
    that occurs in the passed list of events
    """
    result = {}
    for event in events:
        if feature in event:
            if event[feature] in result:
                result[event[feature]].append(event)
            else:
                result[event[feature]] = [event]
    return result

def extends(event, extender): 
    """
    extends(event, extender) :
    returns true if e agrees with event on all the feature--value pairs in event.
    in other words, e is a special case of event with all the information in event
    and perhaps more.  this makes e an extension of event.
    """
    for feature in event:
        if feature not in extender or event[feature] != extender[feature]:
            return False
    return True 

def get_matches(event, events):
    """
    this will return back each of the passed events
    which contain all the information in event 
    (in the sense of extends) 
    """
    return [e for e in events if extends(event, e)]
