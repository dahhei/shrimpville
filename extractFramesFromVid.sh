#!/usr/bin/env bash

#debug stuff
#set -x
#trap read debug


FILE="${1}"

filename="${FILE##*/}"
filenameNoExtension="${filename%%.*}"

outputDir="videos/"${filenameNoExtension}""

[[ -d "${outputDir}" ]] && { printf "\nDir already exists, verify if you want to run by removing dir or changing its name\n" ; exit 1; }

mkdir -p "${outputDir}"

ffmpeg -i "${FILE}" -q:v 2 -start_number 0 "${outputDir}"/'%05d.jpg'
