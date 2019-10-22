import sys
from RLSA import RSLA

rsla = RSLA()
a = open(sys.argv[1])
b = a.read()
c = rsla.annotate(b)
for i in c.views:
    a = i.__dict__
    print (a)
    c = a.get("contains")
    bd = a.get("annotations")
    for d in bd:
        print (d.__dict__)

