import os
import pexpect

TEST_OUTPUT_FILE = "test_output.csv"

if os.name == 'nt':
    EXE = "./covid19ibm.exe"
else:
    EXE = "./covid19ibm"
command = EXE

def main():
    file_output = open(TEST_OUTPUT_FILE, "w")

    print("call C program")
    child = pexpect.spawn(command)
    child.expect("Enter string>")
    child.sendline("Hello")
    result = child.expect("OK")
    output = child.readline()
    print("Result: {}".format(result))
    print("Output: {}". format(output))
    child.close()
    print("done calling C program")

if __name__ == "__main__":
    main()
