import os
import glob
from collections import defaultdict
import time

while(True):
  path = "explore_version_03/checkpoint/*/*.*"
  listFiles=defaultdict()
  for file in glob.glob(path):
    if file.endswith("checkpoint.pth.tar"):
      key=file.split("/")[7]
      if listFiles.get(key) != None :
        listFiles[key].append(file.split("/")[8])
      else:
        listFiles[key] = [file.split("/")[8]]

  basePath="explore_version_03/checkpoint/"


  deleteList=[]
  for k in listFiles.keys():
    checkpoints=listFiles[k]
    checkpoints.sort(reverse=True)
    if len(checkpoints)>=2:
      checkpoints.pop(0)
      for c in checkpoints:
        deleteList.append(basePath+k+"/"+c)


  for d in deleteList:
    print("deletedFiles", d)
    os.remove(d)