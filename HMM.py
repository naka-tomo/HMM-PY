# encoding: utf8
#from __future__ import unicode_literals
import numpy
import random
import math
import time
import codecs
import os
from collections import defaultdict


class HMM(object):

    def __init__(self, nclass):
        self.num_class = nclass
        self.word_class = {}
        self.trans_prob = numpy.ones( (nclass,nclass) )
        self.trans_prob_bos = numpy.ones( nclass )
        self.trans_prob_eos = numpy.ones( nclass )
        self.word_count = [ {} for i in range(nclass) ]
        self.num_words = []

    def load_data(self, filename, delimiter=None ):
        self.num_words = [0]*self.num_class
        self.word_count = [ defaultdict(int) for i in range(self.num_class) ]
        self.sentences = [ line.replace("\n","").replace("\r", "").split(delimiter) for line in codecs.open( filename, "r" , "sjis" ).readlines()]
        self.liks = []

        for sentence in self.sentences:
            for w in sentence:
                c = random.randint(0,self.num_class-1)
                self.word_class[id(w)] = c
                self.word_count[c][w] += 1
                self.num_words[c] += 1


        # 遷移確率更新
        self.calc_trans_prob()

        # 尤度計算
        self.liks.append( self.calc_lik() )

    def calc_output_prob(self, c , w ):
        p = ( self.word_count[c].get(w,0) +  0.1 ) / ( self.num_words[c] + self.num_class * 0.1 )
        return p

    def forward_filtering(self, sentence ):
        T = len(sentence)
        a = numpy.zeros( (len(sentence), self.num_class) )                            # 前向き確率


        for t in range(T):
            w = sentence[t]

            for c in range(self.num_class):
                out_prob = self.calc_output_prob( c , w )

                # 遷移確率
                tt = t-1
                if tt>=0:
                    for cc in range(self.num_class):
                        a[t,c] += a[tt,cc] * self.trans_prob[cc, c]
                    a[t,c] *= out_prob
                else:
                    # 最初の単語
                    a[t,c] = out_prob * self.trans_prob_bos[c]

                # 最後の単語の場合
                if t==T-1:
                    a[t,c] *= self.trans_prob_eos[c]
        return a

    def sample_idx(self, prob ):
        accm_prob = [0,] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i

    def backward_sampling(self, a, sentence, use_max_path=False):
        T = a.shape[0]
        t = T-1

        classes = []

        c = -1
        while True:

            # 状態cへ遷移する確率
            if c==-1:
                trans = numpy.ones( self.num_class )
            else:
                trans = self.trans_prob[:,c]

            if use_max_path:
                c = numpy.argmax( a[t]*trans )
            else:
                c = self.sample_idx( a[t]*trans )

            classes.insert( 0, c )

            t = t-1

            if t<0:
                break

        return classes


    def calc_trans_prob( self ):
        self.trans_prob = numpy.zeros( (self.num_class,self.num_class) ) + 0.1
        self.trans_prob_bos = numpy.zeros( self.num_class ) + 0.1
        self.trans_prob_eos = numpy.zeros( self.num_class ) + 0.1

        # 数え上げる
        for words in self.sentences:
            try:
                # BOS
                c = self.word_class[ id(words[0]) ]
                self.trans_prob_bos[c] += 1
            except KeyError,e:
                # gibss samplingで除かれているものは無視
                continue

            for i in range(1,len(words)):
                cc = self.word_class[ id(words[i-1]) ]
                c = self.word_class[ id(words[i]) ]

                self.trans_prob[cc,c] += 1.0

            # EOS
            c = self.word_class[ id(words[-1]) ]
            self.trans_prob_eos[c] += 1

        # 正規化
        self.trans_prob = self.trans_prob / self.trans_prob.sum(1).reshape(self.num_class,1)
        self.trans_prob_bos = self.trans_prob_bos / self.trans_prob_bos.sum()
        self.trans_prob_eos = self.trans_prob_eos / self.trans_prob_eos.sum()

    def learn(self,use_max_path=False):
        for sentence in self.sentences:

            # 学習データから削除
            for w in sentence:
                c = self.word_class[id(w)]
                self.word_class.pop( id(w) )
                self.word_count[c][w] -= 1
                self.num_words[c] -= 1

            # 遷移確率更新
            self.calc_trans_prob()

            # foward確率計算
            a = self.forward_filtering( sentence )

            # backward sampling
            # classes: 各単語が分類されたクラス
            classes = self.backward_sampling( a, sentence, use_max_path )

            for w,c in zip( sentence, classes ):
                self.word_class[id(w)] = c
                self.word_count[c][w] += 1
                self.num_words[c] += 1

            # 遷移確率更新
            self.calc_trans_prob()

            self.delete_words()


        # 尤度計算
        self.liks.append( self.calc_lik() )

        return

    def delete_words(self):
        for c in range(self.num_class):
            for w,num in self.word_count[c].items():
                if num==0:
                    self.word_count[c].pop( w )

    def save_result(self, dir ):
        if not os.path.exists(dir):
            os.mkdir(dir)

        for c in range(self.num_class):
            path = os.path.join( dir , "word_count_%03d.txt" %c )
            f = codecs.open( path, "w" , "sjis" )
            for w,num in self.word_count[c].items():
                f.write( "%s\t%d\n" % (w,num) )
            f.close()

        path = os.path.join( dir , "result.txt" )
        f = codecs.open( path ,  "w" , "sjis" )
        for words in self.sentences:
            for w in words:
                c = self.word_class[id(w)]
                f.write( str(c)+" " )

            f.write("\n")
        f.close()

        numpy.savetxt( os.path.join(dir,"trans.txt") , self.trans_prob , delimiter="\t" )
        numpy.savetxt( os.path.join(dir,"trans_bos.txt") , self.trans_prob_bos , delimiter="\t" )
        numpy.savetxt( os.path.join(dir,"trans_eos.txt") , self.trans_prob_eos , delimiter="\t" )

        numpy.savetxt( os.path.join(dir,"liks.txt") , self.liks )


    def calc_lik(self):
        lik = 0
        for words in self.sentences:
            # BOS
            w = words[0]
            c = self.word_class[id(w)]
            lik += self.trans_prob_bos[c] * self.calc_output_prob( c, w )

            # 単語間の遷移
            for i in range(1,len(words)):
                w = words[i]
                c = self.word_class[ id(w) ]
                cc = self.word_class[ id(words[i-1]) ]
                lik += self.trans_prob[cc,c] * self.calc_output_prob( c, w )

            # EOS
            w = words[-1]
            c = self.word_class[id(w)]
            lik += self.trans_prob_bos[c] * self.calc_output_prob( c, w )

        return lik

    def get_lik(self):
        return self.liks[-1]



def main():
    hmm = HMM( 3 )
    hmm.load_data( "data.txt" )

    for it in range(10):
        print it,
        hmm.learn()
        print hmm.liks[-1]

    hmm.learn( True )
    hmm.save_result("result")
    return






if __name__ == '__main__':
    main()