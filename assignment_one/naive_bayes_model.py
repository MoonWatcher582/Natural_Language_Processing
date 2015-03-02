import random, collections

class Naive_Bayes_Model(object) :
    def __init__(self, prior, likelihoods):
        """make a naive Bayes model with a specified prior over classes
        and class-dependent likelihood distributions"""
        self.prior = prior
        self.likelihoods = likelihoods
        self.featuredict = {}
    
    def most_informative_features(self, n=100):
        features = set()
        maxprob = collections.defaultdict(lambda: 0.0)
        minprob = collections.defaultdict(lambda: 1.0)
        for cat in self.prior.outcomes():
            for fdist in self.likelihoods[cat]:
                for fval in fdist.outcomes():
                    feature = (fdist.variable, fval)
                    features.add( feature )
                    p = fdist.probability(fval)
                    maxprob[feature] = max(p, maxprob[feature])
                    minprob[feature] = min(p, minprob[feature])
                    if minprob[feature] == 0:
                        features.discard(feature)

        # Convert features to a list, & sort it by how informative
        # features are.
        features = sorted(features, key=lambda feature_: minprob[feature_]/maxprob[feature_])
        return features[:n]

        
    def _cat_prob(self, event, category):
        """compute the probability of the event in this model, 
        assuming that the event actually belongs to the specified category"""
        p = self.prior.probability(category)
        for dist in self.likelihoods[category]:
            if dist.variable in event:
                p = p * dist.probability(event[dist.variable])
        return p
    
    def probability(self, event):
        """compute the probability of an event according to the model, 
        marginalizing over the event category if it's not specified"""
        if self.prior.variable in event:
            category = event[self.prior.variable]
            return self._cat_prob(event, category)
        else:
            p = 0.
            for category in self.prior.outcomes():
                p = p + self._cat_prob(event, category)
            return p
           
    def classify(self, event):
        """classify an event according to its probability in the model"""
        return max(self.prior.outcomes(), 
                   key = lambda c: self._cat_prob(event, c))
    
    def outcomes(self):
        """creates a generator that yields all the (dictionary, probability) pairs
        where dictionary gives a specific event specified by the model 
        and probability gives the probability the model assigns to this event"""
        def last(desc, p):
            return (dict(desc), p)
        def extend(generator, dist):
            for desc, p in generator:
                for o in dist.outcomes():
                    yield ([(dist.variable, o)] + desc, p * dist.probability(o))
        for c in self.prior.outcomes():
            gen = (([(self.prior.variable, o)], self.prior.probability(o)) for o in [c])
            for dist in self.likelihoods[c]:
                gen = extend(gen, dist)
            for desc, p in gen:
                yield last(desc, p)
    
    def accuracy(self, rule):
        """compute the expected accuracy of a specific classification rule
        according to the model.  Rule is a function from events to classes.
        Applies rule to each of the outcomes() of the model, with the true
        category of the outcome hidden, and sums the probability that
        the rule predicts the correct class for that outcome"""
        def correct(event):
            copy = dict(event)
            del copy[self.prior.variable]
            prediction = rule(copy)
            return prediction == event[self.prior.variable]
        return sum(p for (event, p) in self.outcomes() if correct(event))
    
    def optimum(self):
        """return the performance of the best possible classifier
        (namely, the one that uses the true probabilities to make its decision)"""
        return self.accuracy(self.classify)
    
    def sample(self):
        """create a random event using the distribution defined by the model"""
        event = {}
        (k, v) = self.prior.sample()
        event[k] = v
        for dict in self.likelihoods[v]:
            (k, v) = dict.sample()
            event[k] = v
        return event
    
    def __repr__(self) :
        specs = lambda v:''.join([('  %s\n' % dist) for dist in self.likelihoods[v]])
        classes = [('Given %s\n%s' % (v, specs(v))) for v in self.prior.table]
        return "Naive Bayes Model\n%s\n%s" % (self.prior, ''.join(classes))          
