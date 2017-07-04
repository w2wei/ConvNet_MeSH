## set up a new VM
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install htop

## create directories
mkdir exp
mkdir download

## move files to destination directories from idash-data
scp -P 9221 w2wei@192.168.235.51:/home/w2wei/code.tar.gz ~
scp -P 9221 w2wei@192.168.235.51:/home/w2wei/data.tar.gz ~
tar -zxvf code.tar.gz
tar -zxvf data.tar.gz

## install packages
### install anaconda
wget https://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh -P download
sudo bash ~/download/Anaconda2-4.1.1-Linux-x86_64.sh
rm ~/download/Anaconda2-4.1.1-Linux-x86_64.sh
echo "PYTHONPATH=$PYTHONPATH:/usr/local/lib/python2.7/dist-packages">>.bashrc ## add a new python path 
source .bashrc
### Prepending PATH=/home/w2wei/anaconda2/bin to PATH in /home/w2wei/.bashrc
### A backup will be made to: /home/w2wei/.bashrc-anaconda2.bak
### For this change to become active, you have to open a new terminal.

### install nltk punkt
sudo python -m nltk.downloader punkt

### install theano
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
sudo pip install Theano

### install fish
cd download
git clone https://github.com/lericson/fish.git
cd fish
mv README.md README.rst
sudo python setup.py build
sudo python setup.py install
cd ~
# rm -fr ~/download/fish

### install tqdm
cd download
git clone https://github.com/tqdm/tqdm.git
cd tqdm
sudo python setup.py build
sudo python setup.py install
cd ~
# rm -fr ~/download/tqdm
### install pyLucene

