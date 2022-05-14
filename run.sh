export NFS_VOL_NAME=mynfs1
export NFS_LOCAL_MNT=/home
export NFS_SERVER=10.9.84.40
export NFS_SHARE=/mnt/scaleFreeNAS/scaleShare
export NFS_OPTS=vers=4,soft

docker run -u 1015 --gpus all -it --mount \
  "src=$NFS_VOL_NAME,dst=$NFS_LOCAL_MNT,volume-opt=device=:$NFS_SHARE,\"volume-opt=o=addr=$NFS_SERVER\",type=volume,volume-driver=local,volume-opt=type=nfs" \
  ahosny/ml4co
