'''This script runs on 172.21.51.123'''
import os, subprocess

input_dir = os.path.join(os.getcwd(), "nlm2007_input")
output_dir = os.path.join(os.getcwd(),"nlm2007_output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
docList =os.listdir(input_dir)
for doc in docList:
    inFile = os.path.join(input_dir, doc)
    outFile = os.path.join(output_dir,doc[:-2]+"out")
    # metamap -R MSH -N --silent -y
    subprocess.call(['metamap','-N', '-R', 'MSH', '-y', '--silent', inFile, outFile])                                                   