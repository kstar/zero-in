sudo fxload -t fx3 -I /home/akarsh/devel/kde-devel/src/indi-3rdparty/libqhy/firmware/QHY5III178.img -D $(lsusb | grep 'Cypress WestBridge' | cut -d':' -f1 | sed 's:Bus \([0-9]*\) Device \([0-9]*\):/dev/bus/usb/\1/\2:')
lsusb | grep 'QHY'
