#!/usr/bin/env bash

trap "exit" INT TERM
trap "kill 0" EXIT

python rpc.py localhost 7000 m 1 &
python rpc.py localhost 7000 j 1 &

wait %1 %2
