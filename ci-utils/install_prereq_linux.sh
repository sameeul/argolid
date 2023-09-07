#!/bin/bash
# Usage: $bash install_prereq_linux.sh $INSTALL_DIR
# Default $INSTALL_DIR = ./local_install
#
if [ -z "$1" ]
then
      echo "No path to the OmePyrGen source location provided"
      echo "Creating local_install directory"
      LOCAL_INSTALL_DIR="local_install"
else
     LOCAL_INSTALL_DIR=$1
fi

mkdir -p $LOCAL_INSTALL_DIR
mkdir -p $LOCAL_INSTALL_DIR/include

curl -L https://github.com/PolusAI/filepattern/archive/refs/heads/master.zip -o filepattern.zip
unzip filepattern.zip
cd filepattern-master/
mkdir build
cd build
cmake -Dfilepattern_SHARED_LIB=ON -DCMAKE_PREFIX_PATH=../../$LOCAL_INSTALL_DIR -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR ..
make install -j4
cd ../../

