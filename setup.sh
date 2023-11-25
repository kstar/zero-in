if [[ "$1" == '--simulators' ]]; then
    screen -c /home/akarsh/devel/scope_positioning/screenrc-simulators
else
    if [ "$#" -ge 1 ]; then
	>&2 echo "Unrecognized arguments. Usage: $0 [--simulators]"; exit 1;
    fi
    screen -c /home/akarsh/devel/scope_positioning/screenrc
fi
