def Classify_Species(Folder_With_Images, Classifier_Location):
  import torch
  from torchvision.datasets import ImageFolder
  from torchvision.transforms import transforms
  from torch.utils.data import DataLoader
  from torchvision.models import resnet18
  import shutil
  import os
  
  # Define transforms for preprocessing the data
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  
  # Load the saved model
  model = resnet18(pretrained=False)
  num_classes = 5  # Assuming there are 10 classes
  model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Modify the final fully connected layer
  model.load_state_dict(torch.load(Classifier_Location))
  model.eval()
  
  # Load and preprocess the input image
  # Define the dataset
  # Create a temporary directory to organize the images
  temp_dir = Folder_With_Images + '/temp_directory'
  
  os.makedirs(temp_dir, exist_ok=True)
  
  # Move images to the label subdirectory
  for filename in os.listdir(Folder_With_Images):
      if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):  # JPGs or PNGs
          src = os.path.join(Folder_With_Images, filename)
          dst = os.path.join(temp_dir, filename)
          shutil.copyfile(src, dst)
  
  dataset = ImageFolder(root=Folder_With_Images, transform=transform)
  
  # Define the data loader
  batch_size = 32
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  
  predicted_classes = []
  # Perform inference
  with torch.no_grad():
      for images, labels in dataloader:
          outputs = model(images)
          _, predicted = torch.max(outputs, 1)
          predicted_classes.extend(predicted.tolist())
  class_labels = { 0:'cucullata', 1:'longicephala' , 2:'longispina',3:'magna', 4:'pulex'}
  species = [class_labels[class_index] for class_index in predicted_classes]
  
  return species

