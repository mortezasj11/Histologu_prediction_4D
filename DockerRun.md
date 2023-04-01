
```bash 
docker build -t pyradiomics .
```

tensorboard --logdir=runs_NG_focal --port=6012
tensorboard --logdir=runs_NG_all --port=6006

## Docker (personal)
```bash 



docker run -it --rm --gpus all --shm-size=200G --user $(id -u):$(id -g) --cpuset-cpus=175-199 \
-v /rsrch1/ip/msalehjahromi/data/Histology/Nature_Gemini_4d_Feb2022/code:/home/msalehjahromi/code \
-v /rsrch1/ip/msalehjahromi/data:/Data \
--name Hist7 pyradiomics:latest

--gpus '"device=3,4"'

cd /home/msalehjahromi/miniconda/envs/py385/lib/python3.8/site-packages/monai/data
vim utils.py
Esc, Shift+R, typing, Esc, :wq , Enter

  GPU         CPUs

   0           0-24

   1           25-49

   2           50-74

   3           75-99

   4           100-124

   5           125-149

   6           150-174

   7           175-199

   None        200-249


docker run -it --rm --gpus all --shm-size=200G --user $(id -u):$(id -g) --cpuset-cpus=175-199 \
-v /rsrch1/ip/msalehjahromi/data:/Data \
--name Hist000 pyradiomics:latest
