file_path = 'data/raw/SMSSpamCollection'
data_list = []


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

      # preview the result
      for item in data_list[:5]:
          print(f"Token: {item[0]} | Content: {item[1]}")

  except FileNotFoundError:
      print("The file was not found. Check your path:", file_path)

read_data()