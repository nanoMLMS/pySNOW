from snow.io.xyz import read_xyz

from snow.descriptors.cna import *


import numpy as np
from collections import Counter


path='/Users/sofiazinzani/Documents/Dottorato/Unimi/Codici/pySNOW/tutorial/tutorial_structures/Au13_Ih.xyz'

au_lattice=4.08 #AA
el, coords=read_xyz(path)
print(len(el))

c=cna_peratom(1, coords, au_lattice*0.85)

print(c)


from collections import Counter
import numpy as np

# Supponiamo che 'c' sia la lista restituita da cna_peratom
def make_hashable(tup):
    # Converte array in tuple di tuple
    arr1, arr2 = tup
    return (tuple(map(tuple, arr1)), tuple(arr2))

# Converte tutte le tuple
hashable_list = [make_hashable(t) for t in c]
print(hashable_list)
# Conta le occorrenze
counts = Counter(hashable_list)

# Stampa le tuple uniche e il loro conteggio
for item, count in counts.items():
    print("\n\nTuple:\n", item, "\nCount:", count, "\n")



# Numero di tuple uniche
num_unique = len(counts)
print("Numero di tuple uniche:", num_unique)



from collections import Counter

# Converte le sequenze in tuple (hashable) e conta
seq_counts = Counter(tuple(t[1]) for t in c)

# Stampa sequenze e conteggi
for seq, count in seq_counts.items():
    print("Sequenza:", seq, "-> Count:", count)

print("Numero di sequenze uniche:", len(seq_counts))
