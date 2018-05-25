#!/usr/bin/env bash

FILE="imdb_model.h5"
if [ ! -f "$FILE" ]
then
    wget https://github.com/AvantiShri/model_storage/raw/1a7f5e1/abstention/imdb_model.h5
else
    echo "File imdb_model.h5 exists already"
fi
