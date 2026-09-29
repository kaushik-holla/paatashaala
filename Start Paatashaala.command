#!/bin/zsh
cd -- "$(dirname -- "$0")"
/usr/bin/python3 local-setup/manage.py start
