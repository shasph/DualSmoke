import subprocess, sys
import settings
import os
import shutil
import rasterize

argv_1 = ''

def checkStdout(stdout):
    s = str(stdout)
    s = s.split(',')
    if '0' in s[0]:
        return False
    global argv_1
    argv_1 = s[1]
    return True

def terminate():
    shutil.rmtree('tmp')
    sys.exit()

# settings.init()
os.makedirs('tmp', exist_ok=True)
cp = subprocess.run(['python', 'paint.py'], stdout=subprocess.PIPE)
if not checkStdout(cp.stdout):
    print('System Terminated.')
    terminate()
rasterize.rasterize()
cp = subprocess.run(['python', 'client.py'])
cp = subprocess.run(['./build/manta', 'sim.py', argv_1])
terminate()
