# converts the videos mp3
import os
import subprocess

files=os.listdir('videos')
for file in files:
  # print(file)  
  tutorial_number = file.split("Tutorial #")[1].split(" ")[0] # [0] before val [1] after value
  file_name=file.split("  ")[0]
  print(tutorial_number,file_name)
  
  input_file = f"videos/{file}"
  output_file = f"Audio/{tutorial_number}_{file_name}.mp3"

  subprocess.run([
        "ffmpeg",
        "-i", input_file,
        output_file
  ])
  
