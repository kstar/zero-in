#!/bin/bash
cd ~/devel/scope_positioning
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi
./main.py $@
