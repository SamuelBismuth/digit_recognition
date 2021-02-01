#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR/..

sudo docker build -t digit_recognition . 

sudo docker run --rm -it digit_recognition bash