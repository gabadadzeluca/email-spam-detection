file_path = 'data/raw/SMSSpamCollection'
data_list = []

TRAINING_DATA_RATIO = 0.7

def read_data():
  try:
      with open(file_path, 'r') as file:
          for line in file:
              # strip whitespace and skip empty lines
              line = line.strip()
              if not line:
                  continue
              
            #separate the first token from the rest of the line
              parts = line.split(maxsplit=1)
              
              if len(parts) == 2:
                  input_1 = parts[0]
                  input_2 = parts[1]
                  data_list.append((input_1, input_2))
              else:
                  # handle cases where there might only be one token
                  print(f"Skipping malformed line: {line}")


      if(data_list):
          split_index = int(len(data_list) * TRAINING_DATA_RATIO)
          print("SPLIT INDEX: ", split_index)
          train_data = data_list[:split_index] # From start to split_index
          test_data = data_list[split_index:]  # From split_index to the end
          return train_data, test_data
      else:
          return None
      
  except FileNotFoundError:
      print("The file was not found. Check your path:", file_path)
      return None