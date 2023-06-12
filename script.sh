#!/bin/bash
# RUN ME IN {project_folder}/
# bash script.sh

loop_num=10
for ((i=0;i<=10;i++))
do
  python learn_edge.py --prefix ${i}
done
