#!/bin/bash
export DISPLAY=':0'
set -o xtrace
if [ $(uname -m) == "armv7l" ]; then
    >&2 echo "Detected Raspberry Pi"
    VIEWPORT="HDMI-1"
    LAPTOP_DISPLAY="HDMI-2"
    if xrandr -d ${DISPLAY} | grep "^${LAPTOP_DISPLAY}" | grep -q "disconnected"; then
	# Force enable the display so that we can VNC into it
	RESOLUTION_W=1355
	RESOLUTION_H=750
	xrandr -d ${DISPLAY} --newmode $(cvt ${RESOLUTION_W} ${RESOLUTION_H} | grep -v '#' | sed 's:Modeline ::')
	MODENAME=$(cvt ${RESOLUTION_W} ${RESOLUTION_H} | grep -v '#' | sed 's:Modeline ::' | awk '{ print $1 }')
	xrandr -d ${DISPLAY} --addmode ${LAPTOP_DISPLAY} ${MODENAME}
	xrandr -d ${DISPLAY} --output ${LAPTOP_DISPLAY} --primary --mode ${MODENAME} --left-of ${VIEWPORT}
    else
	xrandr -d ${DISPLAY} --output ${LAPTOP_DISPLAY} --primary
    fi
else
    >&2 echo "Detected Laptop"
    VIEWPORT="HDMI2"
    #VIEWPORT="VGA1"
    LAPTOP_DISPLAY="eDP1"
fi
xinput map-to-output "cywy USB2IIC_CTP_CONTROL" "${VIEWPORT}" # Map touch panel on HDMI display to the appropriate output
xinput map-to-output "HAILUCK CO.,LTD USB KEYBOARD Mouse" "${VIEWPORT}"
xrandr --output "${VIEWPORT}" --brightness 1.0 --gamma 0.8 # Reduce the brightness of the display (TODO: Tweak)
/home/akarsh/bin/move_windows_to "${LAPTOP_DISPLAY}"

epv_id=$(wmctrl -l | grep 'Eyepiece View$' | cut -d' ' -f1)

# Read window states (maximized, fullscreen etc.)
states=( $(xprop -id ${epv_id} | grep _NET_WM_STATE | cut -d= -f2 | sed "s/ _NET_WM_STATE_//g;s/,/ /g" | tr [:upper:] [:lower:]) )

# Deactivate window states from right to left (so that fullscreen is removed before maximize)
for (( idx=${#states[@]}-1; idx>=0; idx-- )); do
    wmctrl -i -r ${epv_id} -b remove,${states[idx]};
done

# Move the window to the desired viewport
MVARG=0,$(xrandr | grep "${VIEWPORT}" | sed "s/^.* [0-9]\\+x[0-9]\\++\\([0-9]\\+\\)+\\([0-9]\\+\\) .*$/\\1,\\2/"),-1,-1
wmctrl -i -r ${epv_id} -e ${MVARG} || { >&2 echo "Error moving window ${epv_id} with MVARG=${MVARG}"; }

wmctrl -i -r ${epv_id} -b add,fullscreen
wmctrl -i -r ${epv_id} -b remove,focused

# Restart VNC so it picks up new resolution
if systemctl is-active --quiet x11vnc.service; then
    echo "Going to restart VNC! Please enter your USER password for sudo"
    sudo systemctl restart x11vnc.service
fi
set +o xtrace
