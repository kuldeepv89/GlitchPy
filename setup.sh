#!/bin/bash

moddir=`pwd`
echo "Adding ${moddir} to PYTHONPATH"
export PYTHONPATH=${PYTHONPATH}:${moddir}
