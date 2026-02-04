findneighsideswave is a simple python script that solves the "find neighbor" and "find neighboring sides" problems in a given triangulation avoiding big loops and conditionals.

- "Find neighbor" problem: given a list of n-tuples of vertices of cells in a tiling, what are the neighbors of each cell?
- "Find neighboring sides" problem: given a list of n-tuples of vertices of cells in a tiling whose sides are numbered, which side of each cell is shared with which side of its neighbors?

The script can get memory-intensive, so it is recommended for tilings with a medium number of cells. For a smaller number of cells I recommend findneighsides2, and for a big number of cells try findneighsides (long runtime though! parallelize to liking).
