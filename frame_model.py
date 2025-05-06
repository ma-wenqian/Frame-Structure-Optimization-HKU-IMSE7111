import openseespy.opensees as ops
import numpy as np
import opsvis as opsv
import matplotlib.pyplot as plt
import math
import os

# Units: N, mm, Sec, MPa, 10^3Kg (Tonne)


def frame_model(d, t = np.array([0.05]*8)):
    # a: dimensions of the section, unit: mm
    # t: thickness of the elements as a percentage of the section
    # print("calling frame_model")
    os.chdir(os.path.dirname(__file__))  # set the current working directory to the directory of the script
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    NBays = 3
    NFloors = 4
    WBay = 7000.0
    HStory = np.array([5500.0] + [4000.0] * (NFloors - 1))
    # define material properties
    E = 2.1e5  # MPa
    density = 7.3  # T/m^3
    I = (d**4 - (d-d*t)**4) / 12 # Moment of Inertia, mm^4
    area = d**2 - (d-d*t)**2  # mm^2
    TotalMass = ( np.sum(4 * HStory *area[0:4]) + np.sum(area[4:8]*WBay*3) ) * density * 1e-9  # T
    nodalMass = TotalMass / ((NBays + 1 )* (NFloors - 1))  # T

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
    ops.equalDOF(5, 6, 1)
    ops.equalDOF(5, 7, 1)
    ops.equalDOF(5, 8, 1)
    ops.equalDOF(9, 10, 1)
    ops.equalDOF(9, 11, 1)
    ops.equalDOF(9, 12, 1)
    ops.equalDOF(13, 14, 1)
    ops.equalDOF(13, 15, 1)
    ops.equalDOF(13, 16, 1)
    ops.equalDOF(17, 18, 1)
    ops.equalDOF(17, 19, 1)
    ops.equalDOF(17, 20, 1)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 1, 1, 1)
    ops.fix(3, 1, 1, 1)
    ops.fix(4, 1, 1, 1)

    # define mass and nodal loads
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for nodetag in range(5, 21):
        ops.mass(nodetag, nodalMass,  nodalMass, 0.0)           # unit: T
        ops.load(nodetag, 0.0, -1.0 * nodalMass * 9800, 0.0)    # unit: N

    # define column elements
    elementTag = 1
    PDeltaTransf =1
    ops.geomTransf('PDelta', PDeltaTransf)
    for i in range(1, NBays + 2):
        for j in range(NFloors):
            node1 = i + j*4
            node2 = node1 + 4
            ops.element('elasticBeamColumn', elementTag, node1, node2, area[j], E, I[j], PDeltaTransf)
            elementTag += 1

    # define beam elements
    LinearTransf = 2
    ops.geomTransf('Linear', LinearTransf)
    for i in range(1, NBays + 1):
        for j in range(1, NFloors + 1):
            node1 = i + j*4
            node2 = node1 + 1
            ops.element('elasticBeamColumn', elementTag, node1, node2, area[j+3], E, I[j+3], LinearTransf)
            elementTag += 1
    
    # gravity analysis
    ops.system('BandGeneral')
    ops.constraints('Plain')
    ops.numberer('RCM')
    ops.test('NormDispIncr', 1.0e-6, 10)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 0.1)
    ops.analysis('Static')
    ops.analyze(10)
    # print("Gravity result:",ops.nodeDisp(17,2)) # unit: mm
    return TotalMass


def dynamic_analysis():
    ops.loadConst('-time', 0.0)
    # Eigenvalue Analysis
    nEigenI = 1  # mode i = 1
    nEigenJ = 2  # mode j = 2
    lambdaN = ops.eigen(nEigenJ)  # eigenvalue analysis for nEigenJ modes
    lambdaI = lambdaN[nEigenI - 1]  # eigenvalue mode i = 1
    lambdaJ = lambdaN[nEigenJ - 1]  # eigenvalue mode j = 2
    w1 = math.sqrt(lambdaI)  # w1 (1st mode circular frequency)
    w2 = math.sqrt(lambdaJ)  # w2 (2nd mode circular frequency)
    T1 = 2.0 * math.pi / w1  # 1st mode period of the structure
    T2 = 2.0 * math.pi / w2  # 2nd mode period of the structure
    # print(f"T1 = {T1:.4f} s")  # display the first mode period in the command window
    # print(f"T2 = {T2:.4f} s")  # display the second mode period in the command window

    zeta = 0.02  # percentage of critical damping
    a0 = zeta * 2.0 * w1 * w2 / (w1 + w2)  # mass damping coefficient based on first and second modes
    a1 = zeta * 2.0 / (w1 + w2)  # stiffness damping coefficient based on first and second modes

    # assign damping to frame beams and columns
    # command: 'rayleigh', alpha_mass, alpha_currentStiff, alpha_initialStiff, alpha_committedStiff
    ops.rayleigh(a0, 0.0, a1, 0.0)  # assign mass proportional damping to structure (only assigns to nodes with mass) and assign stiffness proportional damping to frame beams & columns w/out n modifications


    # define ground motion parameters
    patternID = 2  # load pattern ID
    GMdirection = 1  # ground motion direction (1 = x)
    Scalefact = 1.0  # ground motion scaling factor
    
    dt = 0.02   # time step for ground motion
    GMfile = 'elcentro.txt' # name of the ground motion file
    # output_file = 'elcentro_results.txt' # name of the output file in the 'Output' folder
   
    # define the acceleration series for the ground motion
    # syntax: "Series -dt timestep_of_record -filePath filename_with_acc_history -factor scale_record_by_this_amount"
    g = 9800 # gravity constant, Unit: mm/sec^2
    ops.timeSeries('Path', 2, '-dt', dt, '-filePath', GMfile, '-factor', Scalefact * g)

    # create load pattern: apply acceleration to all fixed nodes with UniformExcitation
    # command: pattern('UniformExcitation', patternID, GMdir, '-accel', timeSeriesID)
    ops.pattern('UniformExcitation', patternID, GMdirection, '-accel', 2)

    # define dynamic analysis parameters
    ops.wipeAnalysis()  # destroy all components of the Analysis object
    ops.constraints('Plain')  # how it handles boundary conditions
    ops.numberer('RCM')  # renumber dof's to minimize band-width (optimization)
    ops.system('UmfPack')  # how to store and solve the system of equations in the analysis
    tol=1.0e-6
    ops.test('NormDispIncr', tol, 10)  # type of convergence criteria with tolerance, max iterations
    # ops.test('FixedNumIter',10)
    ops.algorithm('Newton')  # use Newton's solution algorithm: updates tangent stiffness at every iteration
    ops.integrator('Newmark', 0.5, 0.25)  # uses Newmark's average acceleration method to compute the time history
    ops.analysis('Transient')  # type of analysis: transient or static
    
    GMtime = 15 # only the first 15 seconds of the record
    dt_analysis = 0.002  # timestep of analysis
    NumSteps = round(GMtime / dt_analysis)  # number of steps in analysis
    u17 = np.full(NumSteps, np.nan)


    for i in range(NumSteps):
            ok = ops.analyze(1, dt_analysis)  # perform the transient analysis
            if ok != 0:
                print(f"Analysis failed at step {i} and now tyr fixed.")
                ops.test('FixedNumIter',50)
                ok = ops.analyze(1, dt_analysis)
                if ok == 0:
                    print("That worked .. back to regular newton.")
                ops.test('NormDispIncr', tol, 10)
            u17[i] = ops.nodeDisp(17, 1)  # record the displacement of node 17 in the x-direction
    max_u17 = np.max(np.abs(u17))  # find the maximum absolute displacement of node 17 in the x-direction
    # plt.plot(np.arange(0, GMtime, dt_analysis), u17)  # plot the displacement of node 17 in the x-direction
    # plt.title('Node 17 Displacement')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Displacement (mm)')
    return max_u17, np.arange(0, GMtime, dt_analysis), u17

def result_save(d, file_name="result"):
    TotalMass = frame_model(d)
    max_u17, time, u17 = dynamic_analysis()
    print(f"d={d}")
    print("Total Mass (T)\tMax Displacement (mm)")
    print(f"{TotalMass:.4f}\t\t{max_u17:.4f}")
    if not os.path.exists("results"):
        os.makedirs("results")
    resultfile = os.path.join("results", f'{file_name}.txt')
    data = np.column_stack((time, u17))
    np.savetxt(resultfile, data, header="Time (s), Displacement (mm)", comments=f'# d={d}\n# TotalMass={TotalMass}T, max_u17={max_u17}mm\n', delimiter=',')
    print(f"Results saved to {resultfile}")

if __name__ == "__main__":
    print("frame_model.py")
    # pre1 = np.array([150.0]*8) # predesigned, pre1
    # pre2 = np.array([400.0]*8) # predesigned, pre2
    # pre3 = np.array([650.0]*8) # predesigned, pre3

    # result_save(pre1, "pre1")
    # result_save(pre2, "pre2")
    # result_save(pre3, "pre3")

    # show the model
    # opsv.plot_model(False, False) # not show the node tags and element tags
    # opsv.plot_model(True, True) # show the node tags and element tags
    # plt.axis('off')  # Remove axes from the plot
    # plt.show()


