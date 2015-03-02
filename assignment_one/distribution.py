import random, collections

class Distribution(object):
    def __init__(self, variable, table):
        self.variable = variable
        self.table = table
        
    def outcomes(self):
        return (item for item in self.table.keys())
    
    def probability(self, item):
        if item in self.table:
            return self.table[item]
        else:
            return 0.
        
    def sample(self):
        # http://en.wikipedia.org/wiki/Inverse_transform_sampling
        v = random.uniform(0, 1)
        running_cumulative = 0.0
        for item,probability in self.table.items():
            if running_cumulative < v < (running_cumulative + probability): 
                return (self.variable, item)
            running_cumulative += probability
        raise NotImplementedError("The cuumulative should at max be 1, it wasn't, so this failed")
            
    def __repr__(self):
        return "{%s}" % ', '.join([('P(%s=%s)=%s' % (self.variable, v, self.table[v])) for v in self.table])
