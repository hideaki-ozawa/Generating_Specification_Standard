

# Common item
def intersection_1D(list1, list2):
    """
    Return the intersection of 2 one-dimensional list.
    :param list1: [1, 0, 1, 0,...]
    :param list2: [1, 1, 0, 0,...]
    :return: [1, 0, 0, 0,...]
    """
    dim = len(list1)
    intersection = [None]*dim
    for i in range(0, dim):
        intersection[i] = list1[i]*list2[i]
    return intersection


def intersection_ship_matrix(ship_matrix1, ship_matrix2):
    """
    Return the 'intersection' of 2 two-dimensional list.
    :param ship_matrix1: [[1, 0, 1, ...],      section 1
                          [1, 1, ...],         section 2
                          ...
                          [1, 0, ...]]
    :param ship_matrix2: ditto
    :return: [[1, 0, 1, ...],      section 1
              [1, 1, ...],         section 2
              ...
              [1, 0, ...]]
    """
    num_section = len(ship_matrix1)
    intersection = [None]*num_section
    for i in range(0, num_section):
        intersection[i] = intersection_1D(ship_matrix1[i], ship_matrix2[i])
    return intersection


def relative_complement_1D(l1, l2):
    """
    Return the relative complement of 2 one-dimensional list where *l2* is the intersection.
    :param l1: [1, 0, 1, 0,...]
    :param l2: [1, 1, 0, 0,...]
    :return: [0, 0, 1, 0, ...]
    """
    n = len(l1)
    relative_complement = [0]*n
    for i in range(0, n):
        if (l1[i]==1)&(l2[i]==0):
            relative_complement[i] =1
    return relative_complement


def relative_complement_node_matrix(node_matrix1, node_matrix2):
    """
    Return the 'relative complement' of 2 two-dimensional list.
    """
    num_section = len(node_matrix1)
    relative_complement = [None]*num_section
    for i in range(0, num_section):
        relative_complement[i] = relative_complement_1D(node_matrix1[i], node_matrix2[i])
    return relative_complement


def section_matrix2ship_matrix(section_matrix):
    """
    :param section_matrix:  list of outputs of section_matrix().
    :return: [[[1, 1, ...],       [[1, 1, ...],     section 1
                   ...    ,  ...      ...
               [1, 0, ...]],       [1, 0, ...]]]    section 9

                 ship 1               ship n
    """
    num_section = len(section_matrix)
    num_ship = len(section_matrix[0])
    ship_matrix = [None]*num_ship
    for i in range(0, num_ship):
        ship_matrix[i] = [None]*num_section
        for j in range(0, num_section):
            ship_matrix[i][j]=section_matrix[j][i]
    return ship_matrix


def node_name(shipname, Z1):
    """
    Return the name list of nodes in dendrogram.
    :param shipname: ['1727', '0937', 'Z373', ...]
    :param Z1: Z1 = Z.astype(int) where Z is one of the output of linkage().
    :return: ['0937+Z373','1727+0937&Z373', ...]
    """
    n = len(shipname)
    for i in range(0, len(Z1)):
        left = shipname[Z1[i][0]]
        right = shipname[Z1[i][1]]
        left = left.replace('+', '&')
        right = right.replace('+', '&')
        shipname.append(left+'+'+right)
    node_name = shipname[n:]
    return node_name


def node_matrixs(ship_matrixs, Z1):
    """
    Return the node matrices with the same form as the output of section_matrix2ship_matrix().
    :param ship_matrixs: the output of section_matrix2ship_matrix().
    :param Z1: Z1 = Z.astype(int) where Z is one of the output of linkage().
    :return: [[[1, 1, ...],       [[1, 1, ...],     section 1
                   ...    ,  ...      ...
               [1, 0, ...]],       [1, 0, ...]]]    section 9

                 node 1               node n
    """
    n = len(ship_matrixs)
    for i in range(0, len(Z1)):
        matrix1 = ship_matrixs[Z1[i][0]]
        matrix2 = ship_matrixs[Z1[i][1]]
        ship_matrixs.append(intersection_ship_matrix(matrix1, matrix2))
    node_matrixs = ship_matrixs[n:]
    return node_matrixs


def common_item(ship_matrix_section, item_union_section):
    """
    Return the common sentences of the section of specifications and corresponding union set.
    :param ship_matrix_section: [1, 1, 0, ...]
    :param item_union_section: output of section_matrix_and_item_union().
    :return: ['sentence1', 'sentence2', ...]
    """
    common_item = []
    for i in range(0, len(ship_matrix_section)):
        if ship_matrix_section[i]==1:
            common_item.append(item_union_section[i])
    return common_item


def node_matrixs_difference(ship_matrixs, node_matrixs, Z1, LorR):
    """
    Return the relative complement of the node in the dendrogram and its sub-node.
    :param ship_matrixs: output of section_matrix2ship_matrix().
    :param node_matrixs: output of node_matrixs().
    :param Z1: Z1 = Z.astype(int) where Z is one of the output of linkage().
    :param LorR: 'L' or 'R'. 'L' stands for left sub-node and 'R' stands for right sub-node.
    :return: having the same form as the output of node_matrixs().
    """
    n = len(ship_matrixs)
    matrixs = ship_matrixs+node_matrixs
    if LorR =='L':
        for i in range(0, len(Z1)):
            matrix1 = matrixs[Z1[i][0]]
            matrix2 = matrixs[n+i]
            ship_matrixs.append(relative_complement_node_matrix(matrix1, matrix2))
        node_matrixs_difference = ship_matrixs[n:]

    if LorR =='R':
        for i in range(0, len(Z1)):
            matrix1 = matrixs[Z1[i][1]]
            matrix2 = matrixs[n+i]
            ship_matrixs.append(relative_complement_node_matrix(matrix1, matrix2))
        node_matrixs_difference = ship_matrixs[n:]

    return node_matrixs_difference


def different_item(node_matrix_section, item_union_section):
    """
    Return the different sentences between node and its sub-node.
    :param node_matrix_section: [1, 1, 0, ...]
    :param item_union_section: output of section_matrix_and_item_union().
    :return: ['sentence1', 'sentence2', ...]
    """
    different_item = []
    for i in range(0, len(node_matrix_section)):
        if node_matrix_section[i]==1:
            different_item.append(item_union_section[i])
    return different_item



