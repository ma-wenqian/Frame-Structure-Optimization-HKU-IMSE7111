import openseespy.opensees as ops
import numpy as np
import opsvis as opsv
import matplotlib.pyplot as plt
# Units: N, mm, Sec, MPa, 10^3Kg (Tonne)

def frame_model(a = np.array([200.0]*8), t=np.array([20.0]*8)):
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    NBays = 3
    NFloors = 4
    WBay = 6000.0
    HStory = np.array([5500.0] + [4000.0] * (NFloors - 1))
    # define material properties
    E = 2.1e5  # MPa
    density = 7.3  # T/m^3
    area = a**2 - (a-t)**2  # mm^2
    TotalMass = ( np.sum(2 * HStory *area[0:4]) + np.sum(area[4:8]*WBay*3) ) * density * 1e-9  # T
    nodalMass = TotalMass / ((NBays + 1 )* (NFloors - 1))  # 
    print('nodalMass', nodalMass)

    # Calculate the locations of beam-coumn joint centerlines
    Pier = np.array([i * WBay for i in range(NBays + 1)]) 
    Floor = np.array([sum(HStory[:i]) for i in range(NFloors + 1)])



    # define nodes
    nodeTag = 1
    for floor in Floor: 
        for pier in Pier:
            ops.node(nodeTag, pier, floor)
            nodeTag += 1
    # define boundary conditions
    ops.equalDOF(1, 2, 1)




    opsv.plot_model()
    plt.show()


frame_model()

