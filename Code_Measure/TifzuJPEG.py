##### Convert all non jpg to jpg, Ignores files that cant be made to jpg. 

def ConvertTiftoJPEG(directory, save_loc):
  import os as os
  from PIL import Image
  import shutil
  # Input: directory of Iamge folder
  # Savefolder, in whic the JPG folder is created
  save_folder = save_loc +"/JPG"
  if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
  os.mkdir(save_folder)

  for root, dirs, files in os.walk(directory, topdown=False):
    #print(directory)
    for name in files:
      outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
      Image_name = outfile.split("/")[-1]
      try:
        im = Image.open(os.path.join(root, name))
        print ("Generating jpg for %s" )
        im.thumbnail(im.size)
        im.save(save_folder + "/" + Image_name, "JPEG", quality=100)
      except Exception:
        print(f"Could not save {os.path.splitext(os.path.join(root, name))}")



