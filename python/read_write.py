import subprocess, os

run_file = subprocess.Popen("./read_write")
while not (os.path.isfile("./temp_python.txt")):
    pass
# Do some python things pass create temp file for c
string = "write this to temporary file"
with open("./temp_python.txt") as input:
    a = input.read()
print(a)
os.remove("temp_python.txt")

string = "write this to temporary file"
with open("./temp_c.txt", "w") as output:
    output.write(string)

# Check if process has finished
if run_file.poll() is not None:
    print("process has finised")
