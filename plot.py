#!/usr/bin/env python

import argparse
import Gnuplot

# construct arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('files',nargs='+')
args = parser.parse_args()

# add argument names to the current namespace
locals().update(vars(args))

g=Gnuplot.Gnuplot(persist=1)
items=[]
for file in files:
    contents=open(file).readlines()
    data=[map(float,[x[0].split(':')[1],x[1].split(':')[1]]) for x in map(lambda y: y.strip().split('\t'),contents[2:-2])]
    items.append(Gnuplot.Data(data,title=file,with_='linespoints'))
g.plot(*items)
g.interact()
    