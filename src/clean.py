from PIL import Image
import pathlib
import os

data_dir = pathlib.Path('../images')
image_paths = list(data_dir.glob('*/*.jpg'))

for filename in image_paths:
  try:
      img = Image.open(filename)
      img.verify()
      print("verified")
  except (IOError, SyntaxError, OSError) as e:
      print('Bad file:', filename)
      os.remove(filename)
      print("removed")
