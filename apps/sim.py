from numpy.core.numeric import True_
from manta import *
import numpy as np
import os
import datetime
import settings
import sys

DATA_FOLDER = 'tmp/'

IS_INITIAL_SRC_VEL = True
INPUT_GUIDE = True
INPUT_FLUID = False
OUT_VELOCITY = False
OUT_DENSITY = False
OUT_FLAGS = False
OUT_PRESSURE = False
DATA_OUTPUT = OUT_DENSITY | OUT_VELOCITY | OUT_FLAGS | OUT_PRESSURE
USE_SYSTEM_OBSTACLE = False
USE_USER_OBSTACLE = settings.USER_OBSTACLE
ON_OBSTACLE = USE_SYSTEM_OBSTACLE ^ USE_USER_OBSTACLE
OPEN_xX = True
OPEN_yY = False

def SetBoundary():
    bound = sys.argv[1]
    if(not 't' in bound):
        setOpenBound(flags, 0,'Y',FlagOutflow|FlagEmpty) 
    if(not 'b' in bound):
        setOpenBound(flags, 0,'y',FlagOutflow|FlagEmpty) 
    if(not 'r' in bound):
        setOpenBound(flags, 0,'X',FlagOutflow|FlagEmpty) 
    if(not 'l' in bound):
        setOpenBound(flags, 0,'x',FlagOutflow|FlagEmpty)

def InitialVelocityForSmoke():
    np_vel = np.ndarray((1 if dim==2 else res, res, res, 3))
    center=(0.5, 0.1, 0.5)
    radius = 0.06
    scale = 3
    cX = int(center[0] * res)
    cY = int(center[1] * res)
    cZ = int(center[2] * res)
    r = int(radius * res)
    for ix in range(cX - r, cX + r):
        for iy in range(cY - r, cY + r):
            if dim == 2:
                np_vel[0, iy, ix, 0] = 0
                np_vel[0, iy, ix, 1] = scale
                np_vel[0, iy, ix, 2] = 0
            else:
                pass
    return np_vel
       

# dimension
dim = 2

# resolution 
res = settings.RESOLUTION

# vector : (width, height, depth)
gs = vec3(res, res, 1 if dim==2 else res)
# 
s = Solver(name='main', gridSize = gs, dim = dim)
s.timestep = 0.1

# allocate grids
# 各グリッドに入っているもの？ [Flag=fluid, obstacle, open etc...] 
# reference:grid.h l:292
flags = s.create(FlagGrid)
# velocity:vector [MACGrid=Vec3] 
vel = s.create(MACGrid)
# density:scalar [RealGrid = float]
density = s.create(RealGrid)
# pressure:scalar
pressure = s.create(RealGrid)
# obstacle velocity:vector
# mantaMsg('\nsize x : %i' % vel.getSizeX())


# noisefield.h noisefield.cppを使ってノイズを作成？
noise = s.create(NoiseField, loadFromFile=True)
# ノイズのスケール倍率を設定
noise.posScale = vec3(45)
# ノイズの上下限値の有無
noise.clamp = True
# 下限値の設定
noise.clampNeg = 0
#　上限値の設定
noise.clampPos = 1
# ノイズのオフセットの設定
noise.valOffset = 0.75
# ?
noise.timeAnim = 0.2

if(INPUT_FLUID):
    fluidFile = s.create(FlagGrid) 
    fluidFile.load(DATA_FOLDER + '')
    source = Shape(s)
    source.applyToGrid(fluidFile)
else:
    # 流体の入力形状の設定（Cylinder:筒状, Box:箱状,  etc...）
    source = s.create(Cylinder, center=gs*vec3(0.5, 0.1, 0.5), radius=res*0.05, z=gs*vec3(0, 0.02, 0))


# 各グリッドのフラグを領域全体で初期化
flags.initDomain()
# 各グリッドのフラグを埋める
flags.fillGrid()

if (IS_INITIAL_SRC_VEL):
    initialVel = s.create(MACGrid)
    np_initialVel = InitialVelocityForSmoke()
    copyArrayToGridVec3(np_initialVel, initialVel)


if(ON_OBSTACLE):
    if(USE_SYSTEM_OBSTACLE):
        obsPos = vec3(0.5, 0.35, 0)
        obsSize = 0.1
    #     obs = Sphere(parent = s, center = gs*obsPos, radius=res*obsSize)
        obs = Box( parent=s, p0=gs*vec3(0.3,0.3,0.1), p1=gs*vec3(0.7,0.5,0.9)) 
        phiObs = obs.computeLevelset()

    if(USE_USER_OBSTACLE):
        obsFile = s.create(FlagGrid)
        obsFile.load(DATA_FOLDER + 'obs.npz')
        # obs = Shape(s)
        # obs.applyToGrid(obsFile)
        phiObs = obstacleLevelset(obsFile)

    setObstacleFlags(flags=flags, phiObs=phiObs)
    flags.fillGrid()
    # obs.applyToGrid(grid=density, value=0.)

SetBoundary() 

if(DATA_OUTPUT):
    now = datetime.datetime.now()
    dataDir = "results/{0:%Y%m%d_%H%M%S}".format(now)
    os.mkdir(dataDir)
    if(OUT_DENSITY):
        densityDir = dataDir + "/density"
        os.mkdir(densityDir)
    if(OUT_VELOCITY):
        velocityDir = dataDir + "/velocity"
        os.mkdir(velocityDir)
    if(OUT_FLAGS):
        flagsDir = dataDir + "/flags"
        os.mkdir(flagsDir)
    if(OUT_PRESSURE):
        pressureDir = dataDir + "/pressure"
        os.mkdir(pressureDir)


if(INPUT_GUIDE):
    scale_coef = 0.3
    base_dir = './gens/wlcs2vel_ours'
    guideVelocity = s.create(MACGrid)
    guideRegion = s.create(RealGrid)
    frameVelocity = s.create(MACGrid)
    guideVelocity.load(f'tmp/Vel.npz')
    guideRegion.load(f'tmp/LCS.npz')


if(GUI):
    gui = Gui()
    gui.show(twoD=False)
    gui.pause()

for t in range(1000):
# while True:
    # mantaMsg('\nFrame %i' % (s.frame))
    # 流入
    densityInflow(flags=flags, density=density, noise=noise, shape=source, scale=1.0, sigma=0.5)
    # セミラグランジュ移流の実行(密度)
    advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2)
    # セミラグランジュ移流の実行(速度)
    advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=2)
    # 流出セルの削除？ 
    resetOutflow(flags=flags, real=density)

    if(ON_OBSTACLE):
        # 壁に境界条件を設定
        setWallBcs(flags=flags, vel=vel, phiObs=phiObs)

    # 浮力の追加
    if(INPUT_GUIDE):
        frameVelocity.copyFrom(guideVelocity)
        frameVelocity.sub(vel)
        frameVelocity.multConst(vec3(scale_coef, scale_coef, 0))
        # frameVelocity.multConst(vec3(10,10,0))
        addForceField(flags, vel, frameVelocity, guideRegion) 
        
    addBuoyancy(density=density, vel=vel, gravity=vec3(0, -2.5e-2, 0), flags=flags)

    if IS_INITIAL_SRC_VEL:
        setInitialVelocity(flags, vel, initialVel)

    # 圧力の解決
    solvePressure(flags=flags, vel=vel, pressure=pressure)
    # ステップの実行
    s.step()

    if (DATA_OUTPUT):
        if(OUT_VELOCITY):
            velFile = '%s/vel-%s.txt' % (velocityDir, t)
            vel.save(name=velFile)

        if(OUT_DENSITY):
            densityFile = '%s/density-%s.txt' % (densityDir, t)
            density.save(name=densityFile)

        if(OUT_FLAGS):
            flagsFile = '%s/flags-%s.txt' % (flagsDir, t)
            flags.save(name=flagsFile)

        if(OUT_PRESSURE):
            pressureFile = '%s/pressure-%s.txt' % (pressureDir, t)
            pressure.save(name=pressureFile) 