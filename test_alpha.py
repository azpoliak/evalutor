#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import pdb
import cPickle as pickle


def word_matches(h, ref):
    return sum(1 for w in h if w in ref)

def meteor(h, ref, alpha):
    precision = word_matches(h, ref) / float(len(ref))
    recall = word_matches(h, ref) / float(len(h))

    '''
    try:
        meteor_score = (precision * recall) / ((1 - alpha) * recall + alpha * precision)
    except:
        pdb.set_trace()
    '''

    if recall == 0 and precision == 0:
        return 0
    return (precision * recall) / ((1 - alpha) * recall + alpha * precision)




def main():

    '''
    EVALUATOR METHOD
    '''

    def sentences():
            with open('data/hyp1-hyp2-ref') as f:
                for pair in f:
                    yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

    alpha = .01
    
    all_alphas = [0]


    while alpha < 1:

        evaluate_output = ''
        for h1, h2, ref in islice(sentences(), None):
            rset = set(ref)
            #determine whether to run METEOR or Adam Lopez's starter code
            h1_match = meteor(h1, rset, alpha) #word_matches(h1, rset)
            h2_match = meteor(h2, rset, alpha) #word_matches(h2, rset)
            evaluate_output += (str(1) if h1_match > h2_match else # \begin{cases}
                    (str(0) if h1_match == h2_match
                        else str(-1))) + '\n' # \end{cases}

        evaluate_output = evaluate_output.split('\n')
        (right, wrong) = (0.0,0.0)
        conf = [[0,0,0] for i in xrange(3)]
        for (i, (f_e_r, sg, sy)) in enumerate(zip(open('data/hyp1-hyp2-ref'), open('data/dev.answers'), evaluate_output)):
          try:  
            (g, y) = (int(sg), int(sy))
          except:
            pdb.set_trace()
          conf[g + 1][y + 1] += 1
          if g == y:
            right += 1
          else:
            wrong += 1

        acc = right / (right + wrong)
        
        all_alphas.append(acc)

        alpha += .01

    pickle.dump(all_alphas, open( "all_alphas_01.p", "wb" ) )

if __name__ == '__main__':
    main()