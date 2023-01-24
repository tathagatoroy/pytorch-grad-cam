#!/bin/bash
# a bash script to transfer data from share3 to scratch

#delete exsiting folder
rm -rf /scratch/tathagato
mkdir /scratch/tathagato
rsync -azh --info=progress2 tathagato@ada.iiit.ac.in:/share3/tathagato/DREYEVE_DATA.zip /scratch/tathagato
unzip /scratch/tathagato/DREYEVE_DATA.zip -d /scratch/tathagato

