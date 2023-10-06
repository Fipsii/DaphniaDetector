##### Read in jsonfile and build pandas array
##### We also performe bounding box postprocessing
##### Setting the bounding box so that it always contains
##### Spina tip if detected


def JsonToMeasurement(Annotation_file):
  import json as json
  import pandas as pd
  import numpy as np
  # Input: Annotations of bbox detection
  # Output: Csv with Annotations and 
  # postprocessed Daphnid box
  
  data = json.load(open(Annotation_file))
  ann = data["annotations"]
  Image_Ids = data["images"]
  
  Label_Ids = data["categories"]
  Imageframe = pd.DataFrame(Image_Ids)
  
  Labelframe = pd.DataFrame(Label_Ids)
  Annotationframe = pd.DataFrame(ann)

  ### Make ids into a readable format in the csv

  for x in range(1,len(Labelframe["name"])+1):
    Annotationframe["category_id"] = Annotationframe["category_id"].replace(x, Labelframe["name"][x-1])
  
  ### Same for Images ids
  ### Due to duplicating our images we get image Ids higher than the length of Imageframe
  ### We match the id with the corresponding image name and replace it with the name of the image
  
  
  for x in list(set(Annotationframe["image_id"].tolist())):
    Id_replace = Imageframe["file_name"].loc[Imageframe["id"] == x].tolist()[0]
    Annotationframe["image_id"] = Annotationframe["image_id"].replace(x, Id_replace)
    
  
  #### Dewrangle the coordinates

  Annotationframe[['Xmin','Ymin','bboxWidth','bboxHeight']] = pd.DataFrame(Annotationframe.bbox.tolist(), index= Annotationframe.index)

  ### Make useful columns into a new data frame
  
  SmallFrame = Annotationframe[["id", "image_id","category_id","area","Xmin","Ymin","bboxWidth","bboxHeight"]]
  
  #### Make everything to one row per individual
  count = 0
  for y in Labelframe["name"]:
    temp = SmallFrame[SmallFrame["category_id"] == y] 
    temp.columns = ['id_'+ str(y), 'image_id', 'category_id_'+ str(y),'area_'+ str(y),'Xmin_'+ str(y),'Ymin_'+ str(y),'bboxWidth_'+ str(y),'bboxHeight_'+ str(y)]
    count += 1
    
    if count == 1:
      CompleteDataframe = temp
    else:
      CompleteDataframe = pd.merge(CompleteDataframe, temp, on = ["image_id"], how = "outer")
  
  CompleteDataframe.drop_duplicates(subset=["image_id"], keep='first', inplace=True, ignore_index=True)
  CompleteDataframe.insert(3, 'Imagewidth', Imageframe['width']) 
  CompleteDataframe.insert(4, 'Imageheight', Imageframe['height']) 
  
  List_image_id = []
  ### save only the name of the image not the path as image_id
  for x in range(len(CompleteDataframe["image_id"])):
    List_image_id.append(CompleteDataframe["image_id"][x].split("/")[-1])
  
  ## Perfrom postprocessing
  
  ## We want Spina tip to always be contained in Daphnid box, which is often not the case
  ## So we calculate the X,Y coordinates for the spina
  
  Xmin_St = CompleteDataframe['Xmin_Spina tip'].to_numpy()
  Ymin_St = CompleteDataframe['Ymin_Spina tip'].to_numpy()
  Width_St = CompleteDataframe['bboxWidth_Spina tip'].to_numpy()
  Height_St = CompleteDataframe['bboxHeight_Spina tip'].to_numpy()
  
  Xmin_Daphnid = CompleteDataframe['Xmin_Daphnid'].to_numpy()
  Ymin_Daphnid = CompleteDataframe['Ymin_Daphnid'].to_numpy()
  Width_Daphnid = CompleteDataframe['bboxWidth_Daphnid'].to_numpy()
  Height_Daphnid = CompleteDataframe['bboxHeight_Daphnid'].to_numpy()
  
  ## Get max data 
  
  Xmax_St = np.add(Xmin_St, Width_St)
  Ymax_St = np.add(Ymin_St, Height_St)
  
  Xmax_Daphnid = np.add(Xmin_Daphnid, Width_Daphnid)
  Ymax_Daphnid = np.add(Ymin_Daphnid, Height_Daphnid)
  
  ## No we want to check if the coordinates are inside the area of the Daphnid
  ## Unelegante but does the Job ### What do with nan
  
  for coordinate in range(len(Xmax_St)):
    
    ### Check if we have values that are not float. We only have to check on variable per box
    ### to see if it exists
    if (isinstance(Xmin_St[coordinate], float) == True) and (isinstance(Xmin_Daphnid[coordinate], float) == True):
      
      if Xmin_St[coordinate] < Xmin_Daphnid[coordinate]:
        Xmin_Daphnid[coordinate] = Xmin_St[coordinate] - 10 ## buffer 
        
      if Ymin_St[coordinate] < Ymin_Daphnid[coordinate]:
        Ymin_Daphnid[coordinate] = Ymin_St[coordinate] -10
        
      if Xmax_St[coordinate] > Xmax_Daphnid[coordinate]:
        Xmax_Daphnid[coordinate] = Xmax_St[coordinate] + 10
        
      if Ymax_St[coordinate] > Ymax_Daphnid[coordinate]:
        Ymax_Daphnid[coordinate] = Ymax_St[coordinate] + 10
  
  ### The newly acquired bounding boxes for the body have to be recalualted
  
  New_Width = np.subtract(Xmax_Daphnid, Xmin_Daphnid)
  New_Height = np.subtract(Ymax_Daphnid, Ymin_Daphnid)
  
  CompleteDataframe["bboxWidth_Daphnid"] = pd.Series(New_Width)
  CompleteDataframe["bboxHeight_Daphnid"] = pd.Series(New_Height)
  CompleteDataframe["Xmin_Daphnid"] = pd.Series(Xmin_Daphnid)
  CompleteDataframe["Ymin_Daphnid"] = pd.Series(Ymin_Daphnid)
  
  
  ## Save the data
  CompleteDataframe["image_id"] = List_image_id
  CompleteDataframe.to_csv(Annotation_file[:-5] + ".csv", index = False)

  return CompleteDataframe



def AddSegment(Annotation_file_path):
  import json
  # Iterate through all annotations and add the "segmentation" field
  with open(Annotation_file_path, "r") as Annotation_file:
    annotations = json.load(Annotation_file)

  for annotation in annotations:
    annotation["segmentation"] = []

  with open(Annotation_file_path, "w") as output_file:
    json.dump(annotations, Annotation_file_path, indent=4)

  print(f"Modified annotations saved to {Annotation_file_path}")


def AddSegment(Annotation_file_path):
    import json

    # Iterate through all annotations and add the "segmentation" field
    with open(Annotation_file_path, "r") as Annotation_file:
        annotations = json.load(Annotation_file)

    ### Change the dict on the key annottations in which we add the key "segmentation" with value []   
    for x in annotations['annotations']:
        x['segmentation'] = []

    with open(Annotation_file_path, "w") as output_file:
        json.dump(annotations, output_file, indent=4)  # Fixed the indentation here

    print(f"Modified annotations saved to {Annotation_file_path}")

