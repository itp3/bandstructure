import sys
sys.path.append("../")

from bandstructure.system.tightbinding import TightBinding
from bandstructure.lattice.square import Square

l = Square()
s = TightBinding(l)
s.setParams({'t': 3})

print("Parameters:")
s.showParams()
