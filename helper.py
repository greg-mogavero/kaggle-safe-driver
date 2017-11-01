import numpy as np

def gini_score(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
 
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
    
def gini_normalized(a, p):
    return gini_score(a, p) / gini_score(a, a)

# Returns True if s ends with postfix, else False
def ends_with(s, postfix):
    assert(type(s) is str and type(postfix) is str)
    if len(postfix) > len(s):
        return False
    start = len(s) - len(postfix)
    return True if s[start:] == postfix else False