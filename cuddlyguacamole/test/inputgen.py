


def generate_test_system(dim)
    g = [0,1,2,3]
    x,y,z = np.meshgrid(g,g,g)
    xyz = np.vstack((x.flat, y.flat, z.flat)).T
    xyz = np.ascontiguousarray(xyz)