1. Properties

--------------

Abstract classes for building graph representations consist with ``pytorch-geometric``.

All the Graph in this part should return data as following:

Each Graph data (for each structure):

``x``: Node feature matrix. np.ndarray, with shape [num_nodes, nfeat_node]

``pos``: Node position matrix. np.ndarray, with shape [num_nodes, num_dimensions]

``energy``: target. np.ndarray, shape (1, num_target) , default shape (1,)

``y``: target 1. (alias energy) np.ndarray, shape (1, num_target) , default shape (1,)

``z``: atom numbers (alias atomic_numbers). np.ndarray, with shape [num_nodes,]

``cell``: cell matrix. np.ndarray, with shape [3, 3]

``state_attr``: state feature. np.ndarray, shape (1, nfeat_state)

``edge_index``: Graph connectivity in COO format. np.ndarray, with shape [2, num_edges] and type torch.long

                It is neighbor_index and center_index sequentially.

``edge_weight``: Edge feature matrix. np.ndarray, with shape [num_edges,]

``distance``: (alias edge_weight) distance matrix. np.ndarray, with shape [num_edges, 3]

``edge_attr``: Edge feature matrix. np.ndarray, with shape [num_edges, nfeat_edge]

``distance_vec``: (alias edge_attr) distance 3D matrix (x,y,z). np.ndarray, with shape [num_edges, 3]

``cell_offsets``: offset matrix. np.ndarray, with shape [3, 3]

``force``: force matrix per atom. np.ndarray, with shape [num_nodes, 3]

``stress``: stress matrix per atom. np.ndarray, with shape [6,]

``natoms``: (int) number of atom.

``tags``: tags, not used.

``fixed``: fixed atoms, tags , not used.

2. Usage

--------------

For some general functions in ``function``.
such as for manipulate function for data.
It could be used in:

1. data preprocessing part (mainly numpy function);
2. data transform part (tensor function, and be warped as Transformer class);
3. network part (tensor function)

The 2 part are used but not concern gradient.
The 3 part are used gradient.

a. We strongly recommend in 1 part, do just raw data preparation.

b. We strongly recommend in 2 part, just get edge_index,
edge_weight, edge_attr (if regress_force not concerned),
else do just get edge_index (if regress_force concerned).

c. For speed up, we suggest get edge_index in 1 or 2,
then get edge_weight, edge_attr in 3 network.


