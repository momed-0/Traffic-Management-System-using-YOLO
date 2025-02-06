# Ensure the script is run with sudo
if [ "$EUID" -ne 0 ]; then
	echo "Please run as roo"
	exit 1
fi

sudo jetson_clocks

export DISPLAY=:0
