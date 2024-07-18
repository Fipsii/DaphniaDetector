
def ConfigImport(file_path):
  ## Import yaml file and return the config as dict
  
  from ruamel.yaml import YAML
  yaml=YAML(typ='safe')
  with open(file_path, "r") as stream:
    # default, if not specfied, is 'rt' (round-trip)
    configurations = yaml.load(stream)
  
  return configurations
  

def update_yaml_setting(file_path, setting_key, new_value):
    
    ## Import yaml and change attributes
    ## Only needed if we to check data

    from ruamel.yaml import YAML

    yaml=YAML(typ='rt')
    yaml.preserve_quotes = True

    with open(file_path) as stream:
        data = yaml.load(stream)
    # Update the setting in the loaded data
    data[setting_key] = new_value

    with open(file_path, 'w') as file:
        yaml.dump(data, file)
