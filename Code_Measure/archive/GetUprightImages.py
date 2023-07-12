##### This script reads Annotations of the duplicates and throws out the entries
##### and Images which are non optimal and adds segmentations:[] to json for cvat


### This version of JSON to Measurement does the same as the original: aka
### we create a pandas dataframe, which we will use to create a list of images
### with the minimal bboxes to filter and changes the annotationfile
### We also do not save that dataframe
def JsonToMeasurement_NoSave(Annotation_file):
  import json as json
  import pandas as pd
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

  for x in range(1,len(Imageframe["file_name"])+1):
    Annotationframe["image_id"] = Annotationframe["image_id"].replace(x, Imageframe["file_name"][x-1])
  
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
  
  ## save the data
  CompleteDataframe["image_id"] = List_image_id
  CompleteDataframe.to_csv(Annotation_file[:-5] + ".csv", index = False)
  #print(CompleteDataframe.head())
  return(CompleteDataframe)

def CleanDataPD(data):
  
  ## This functions detects the smallest bbox per duplicate for the body
  ## And returns its name for filtering in the JSON
  import pandas as pd
  Sorted_by_name = data.sort_values(by=["image_id"])
  List_of_splits = Sorted_by_name["image_id"]
  List_of_Org_names = []
  #### To find the org Image we remove the end added by the
  #### turning of the images: _###_deg.jpg; r"_-?\d+_deg\.jpg",
  import re
  
  pattern = r"_-?\d+_deg\.jpg"
  Sorted_by_name["Org_Image"] = Sorted_by_name["image_id"].apply(lambda x: re.sub(pattern, "", str(x)))
  print(len(set(Sorted_by_name["Org_Image"])))
  No_duplicates_df = pd.DataFrame()
  Unique_Values = pd.unique(Sorted_by_name["Org_Image"]).tolist()
  Min_Image_Id_list = []
  
  for Org_Images in Unique_Values:
    Body_areas = Sorted_by_name["bboxWidth_Body"].loc[Sorted_by_name["Org_Image"] == Org_Images]
    
    ## Not detected boxes could be the smallest value
    ## thats why we drop 0's from the list
    Body_areas_non_zero = [i for i in Body_areas if i != 0]
    Body_areas_non_nan = [i for i in Body_areas if pd.isna(i) != True]
    Smallest_Box = min(Body_areas_non_nan) 
    
    if not Smallest_Box: ### if we only find na values for all 11 images just take one image
      #### As we dropped Na and 0s we would expect to yield an empty list which in python is considered 
      #### a false
      Min_Image_Id = Sorted_by_name.loc[(Sorted_by_name["Org_Image"] == Org_Images),"image_id"][0]
    else:
      ### Susbsequently add entries with minimal area and OrgImage to the df
      Min_Image_Id = Sorted_by_name.loc[(Sorted_by_name["Org_Image"] == Org_Images) 
      & (Sorted_by_name["bboxWidth_Body"] == Smallest_Box),"image_id"]
      ### If we do not detect a body box we need the except statement which fills in a 0 value
    
    Min_Image_Id_list.append(Min_Image_Id.iloc[0])
    
  ### Now we want to drop out all COCO entries that do not have our names
  return Min_Image_Id_list

def CleanJSON(annotations, selected_images, all_images, saveloc):
  
  import json as json
  import pandas as pd
  ## open JSON
  with open(annotations, "r") as data_file:
    j_data = json.load(data_file)
    list_of_image_ids = []
    print(len(all_images))
    ## Select only IDs which are in our selected images names
    for names in range(len(all_images)):
      for entry in selected_images:
        #print(entry)
        if j_data['images'][names]['file_name'] == entry: ## this condition is only executed 248 times. Why?
          #print(j_data['images'][names]['file_name'])
          list_of_image_ids.append(j_data['images'][names]['id'])
          
  print(len(list_of_image_ids))  
  No_img_Duplicates = []
  No_anno_Duplicates = []
  ### rewrite the JSON
  for values in list_of_image_ids:
    No_anno_Duplicates.append([annotation for annotation in  j_data["annotations"] if annotation['image_id'] == values])
    No_img_Duplicates.append([annotation for annotation in  j_data["images"] if annotation['id'] == values])

  ## Flatten the data
  No_anno_Duplicates_flat = [item for sublist in No_anno_Duplicates for item in sublist] 
  No_img_Duplicates_flat = [item for sublist in No_img_Duplicates for item in sublist]  
  ## This is the annotations tab in the data

  ## We still need to change the images as well as add segmentation to
  for x in range(len(No_anno_Duplicates_flat)):
    No_anno_Duplicates_flat[x]['segmentation'] = []
  
  j_data["images"] = No_img_Duplicates_flat
  j_data["annotations"] = No_anno_Duplicates_flat

  with open(saveloc + "/Unique_Values.json", "w") as outfile:
    json.dump(j_data,outfile)
  print("JSON cleaned")
  return(j_data)

def CleanJPEGS(Image_Path, list_of_names, saveloc):
  
  ######################################################################
  # Input:                                                             #
  # Image_Path: Path to every image                                    #
  # list_of_names: Images with the smallest bounding box               #
  # saveloc: folder for saving; is created should not exist before     #
  #                                                                    #
  # Output: Saved folder withthe Images with smallest bouding box      #
  ######################################################################
  
  import cv2 as cv2
  import os
  
  Split_Names = []
  os.mkdir(saveloc) ## Should be declared in yaml will be used for further analysis
  
  for Images in range(len(Image_Path)): ## for all images
    
    Name = str.split(Image_Path[Images], "/")[-1]
    
    if Name in list_of_names: ###
      img = cv2.imread(Image_Path[Images])
      cv2.imwrite(saveloc +"/"+ Name, img)
      Split_Names.append(Name)
  
  print("JPGES cleaned")

def Images_list(path_to_images):
  import os as os
  PureNames = []
  Image_names = []
  for root, dirs, files in os.walk(path_to_images, topdown=False):
    #print(dirs, files)
    for name in files:
      #print(os.path.join(root, name))
      Image_names.append(os.path.join(root, name))
      PureNames.append(name)
      #print(files)
  return Image_names, PureNames

