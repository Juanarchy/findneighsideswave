import cupy as np

"""
This script loads a shape (nElems,3) array from a text file whose i-th row contains the index of the (possibly 1-indexed) nodes at each vertex of the i-th triangle 
of a non-ramified triangulation, and saves 3 text files containing each:
    - A shape (nElems,3) array whose i-th row contains, at position j, the index of the cell that neighbors cell i from side j. (ElemNeighsFound.txt)
    - A shape (nElems,3) array whose i-th row contains, at position j, the limiting side of the cell that neighbors cell i from side j. (ElemNeighSidesFound.txt)
    - A shape (nElems,3) array whose i-th row contains the index of the (now 0-indexed) nodes at each vertex of the i-th triangle. (ElemNodesNew.txt)
                 X
                / \ 
               /   \ 
              /     \ 
     X-------X-------X 
      \     / \  1  / \ 
       \   /1n0\2m0/   \      In this example, cell n has cell m as a neighbor from its side 0, while cell m has cell n as a neighbor from
        \ /  2  \ /     \     its side 2. Therefore: 
         X-------X-------X 
        / \     / \               - ElemNeighs[n,0] == m
       /   \   /   \              - ElemNeighs[m,2] == n
      /     \ /     \             - ElemNeighSides[n,0] == 2
     X-------V-------X            - ElemNeighSides[m,2] == 0
     

Its implemented wave/paintdrop-strategy algorithm starts the "wave" at a random cell and compares the three sides of each "wavefront" cell to the three sides of each "outside"
cell, and then positions neighbors and sides accordingly through precise array manipulation thus avoiding long loops and conditional statements of any kind. This means that it 
can efficiently be parallelized or ran on GPUs through CUDA implementations of numpy such as cupy. This algorithm can be easily extended to arbitrary (even higher-dimensional)
non-ramified tilings by translating "sides" into "faces" and all their concerning instructions.

This implementation is a memory-wise middle ground between iterating over each cell (findneighsides) and comparing all cells at once (findneighsides2), and is able to solve
relatively big problems faster without problems.

It also has the option to load a text file containing the indices of cells to remove from the tiling, but I haven't tested what happens if one removes enough cells
to leave nodes isolated. I suspect that would be fine, but the resulting answers would assume the existence of said unused, isolated nodes.

-Juan Andrés Fuenzalida A. Contact: juan[dot]fuenzalidaa[at]sansano[dot]usm[dot]cl
"""

elements=np.loadtxt("ElemNodesMatlab.txt",dtype=int)#load nodes faces array
elements=elements-1 #turn into 0-index if 1-indexed
bad_cells=None

#bad_cells=np.loadtxt("bad_cells.txt",dtype=int) #Bad cells to further remove

if bad_cells is not None:
    elements=np.delete(elements,bad_cells,0)

cedge=np.array([np.random.randint(0,len(elements))],dtype=int)
checked=np.array([],dtype=int)

list_of_cells=np.array(range(len(elements)))

ElemNeighs=np.tile(np.array((-1,-1,-1)),(len(elements),1))
ElemNeighSides=np.tile(np.array((-1,-1,-1)),(len(elements),1))

while not (np.isin(list_of_cells,checked)).all():
    checked=np.hstack((checked,cedge))
    remaining = np.setdiff1d(list_of_cells,checked)
    nedge=np.array([],dtype=int)

    print(f"Checked {len(checked)}/{len(elements)} ({len(checked)/len(elements)*100:.2f}%) cells...")

    eside0=np.sort(np.take(elements[cedge],(0,1),axis=1),axis=1)
    eside1=np.sort(np.take(elements[cedge],(1,2),axis=1),axis=1)
    eside2=np.sort(np.take(elements[cedge],(2,0),axis=1),axis=1)
    oside0=np.sort(np.take(elements[remaining],(0,1),axis=1),axis=1)
    oside1=np.sort(np.take(elements[remaining],(1,2),axis=1),axis=1)
    oside2=np.sort(np.take(elements[remaining],(2,0),axis=1),axis=1)

    e,o=np.meshgrid(np.array(range(len(cedge))),np.array(range(len(remaining))),sparse=True)

    comp=np.where((eside0[e]==oside0[o]).all(axis=2))
    te=cedge[comp[1]]
    to=remaining[comp[0]]

    nedge=np.hstack((nedge,to))

    ElemNeighs[te,0]=to
    ElemNeighSides[te,0]=0

    comp=np.where((eside0[e]==oside1[o]).all(axis=2))
    te=cedge[comp[1]]
    to=remaining[comp[0]]

    nedge=np.hstack((nedge,to))

    ElemNeighs[te,0]=to
    ElemNeighSides[te,0]=1

    comp=np.where((eside0[e]==oside2[o]).all(axis=2))
    te=cedge[comp[1]]
    to=remaining[comp[0]]

    nedge=np.hstack((nedge,to))

    ElemNeighs[te,0]=to
    ElemNeighSides[te,0]=2

    comp=np.where((eside1[e]==oside0[o]).all(axis=2))
    te=cedge[comp[1]]
    to=remaining[comp[0]]

    nedge=np.hstack((nedge,to))

    ElemNeighs[te,1]=to
    ElemNeighSides[te,1]=0

    comp=np.where((eside1[e]==oside1[o]).all(axis=2))
    te=cedge[comp[1]]
    to=remaining[comp[0]]

    nedge=np.hstack((nedge,to))

    ElemNeighs[te,1]=to
    ElemNeighSides[te,1]=1

    comp=np.where((eside1[e]==oside2[o]).all(axis=2))
    te=cedge[comp[1]]
    to=remaining[comp[0]]

    nedge=np.hstack((nedge,to))

    ElemNeighs[te,1]=to
    ElemNeighSides[te,1]=2

    comp=np.where((eside2[e]==oside0[o]).all(axis=2))
    te=cedge[comp[1]]
    to=remaining[comp[0]]

    nedge=np.hstack((nedge,to))

    ElemNeighs[te,2]=to
    ElemNeighSides[te,2]=0

    comp=np.where((eside2[e]==oside1[o]).all(axis=2))
    te=cedge[comp[1]]
    to=remaining[comp[0]]

    nedge=np.hstack((nedge,to))

    ElemNeighs[te,2]=to
    ElemNeighSides[te,2]=1

    comp=np.where((eside2[e]==oside2[o]).all(axis=2))
    te=cedge[comp[1]]
    to=remaining[comp[0]]

    nedge=np.hstack((nedge,to))

    ElemNeighs[te,2]=to
    ElemNeighSides[te,2]=2

    cedge=nedge

#save results
np.savetxt('ElemNeighsFound2.txt',ElemNeighs,fmt="%1d")
np.savetxt('ElemNeighSidesFound2.txt',ElemNeighSides,fmt="%1d")
np.savetxt('ElemNodesNew2.txt',elements,fmt='%1d')