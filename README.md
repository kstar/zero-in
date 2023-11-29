# Zero-In

## Introduction

Zero-in is a proof-of-concept for an advanced plate-solving push-to
system for amateur telescopes that I wrote as a personal project for
my own use.

The software estimates the position of the telescope by combining
information from a camera looking at the sky and inertial motion
sensors (similar to the ones used in all commercial smartphones
today). The software is designed to work with or without an equatorial
platform and with no mechanical alignment of finder scope or motion
sensor required.

This is not a release, it is a proof-of-concept.

Because this is a proof-of-concept, it requires significant technical
skill to set up and use this system. As of this writing, my intention
is to provide this as a reference for the novel alignment and
calibraton algorithms implemented here as well as a proof-of-concept
for what a system focused on catering to the needs of advanced
deep-sky observers might look like.

## What is plate-solving?

Plate-solving refers to the idea of comparing a photograph of the sky
against a database of known patterns of stars to identify a mapping
between pixels and positions on the celestial sphere (i.e. celestial
coordinates). A [landmark paper in this
domain](https://iopscience.iop.org/article/10.1088/0004-6256/139/5/1782/meta)
was associated with the creation of
[astrometry.net](https://astrometry.net) by Dustin Lang et. al. This
project uses the binary form of astrometry.net engine. Please see the
rpi_setup_readme.org file for how to set up a Raspberry Pi
appropriately with astrometry.net in the correct configuration

## What hardware do I need?

1. An electronic finder scope, consisting of a camera and a lens to
   image the sky.

2. A 3-axis motion sensor that can output a quaternion describing its
   orientation.

3. A way to mount these on the telescope, with the finder scope
   pointing roughly in the direction of the telescope

4. A computer. I have gotten it working on a Raspberry Pi but was
   unhappy with the stability. I just settled to pulling USB cables to
   my laptop. It's messy but it works fine and lets me get on with
   deep-sky observing.

For the electronic finder, I specifically used a ZWO ASI 290MC-Cool
camera (way overkill) coupled to an Orion 70mm multi-use finder scope
(also overkill).

For the motion sensor, I used an Arduino Uno board with an Arduino
9-axis motion sensor shield built around BNO055 (with the standard
libraries that come with it).

## What is the advantage of plate solving?

It is independent of mechanical errors in the construction of the
telescope. The only things that can affect it is the relative flexure
of the electronic finder scope with respect to the main telescope and
the temperature drift. This way, complex pointing models are not
needed.

## How do you avoid all mechanical alignments?

1. For the finder-scope to telescope alignment, a single-star
   calibration is needed. I center a star in my telescope, and simply
   tell the software which star I have centered. The alignment process
   takes a plate solve and stores the offset RA/Dec from the plate
   center to the star, and this is good enough. I only need to
   "re-align" when I recollimate, and this is even simpler because of
   the interface.

2. Motion sensor calibration is automatic. Every time the system gets
   appropriate plate-solve results, it will automatically recalibrate
   the motion sensor orientation. I designed this so that you can
   literally duct tape the motion sensor to the telescope -- if the
   tape gives, you can just tape the motion sensor back and the system
   will recalibrate within a few plate-solves!

## Does it handle an equatorial platform / untracked / alt-az tracked scope?

Yes. It handles all of these.

The coordinate mathematics are all in the code. To handle an
equatorial platform, the code recognizes that:

1. The telescope's alt/az axes are frozen in an alt/az frame whose LST
   is equal to the LST when the platform becomes flat.

2. The motion sensor has a sense of true altitude from the
   accelerometer's sense of gravity.

So with appropriate coordinate transforms, an equatorial platform is
easily handled -- the software only needs to know the travel time of
the platform and needs to be informed every time the platform is
reset.

## Okay, walk me through the software stack

The UI is written using the [Qt
Framework](https://www.qt.io/product/framework) through
`PyQt5`. Communications with the camera hardware is handled using
[INDI](http://indilib.org/) via `PyINDI`. Some coordinate math is
implemented by hand where simplicity and speed are of the
essence. Others are sourced from `astropy`. Python libraries are used
heavily wherever available to simplify the implementation. The
plate-solving itself is done by invoking `astrometry.net`'s
`solve-field` binary, but the source extraction is done using
[SEP](https://sep.readthedocs.io/en/v1.1.x/).

The system is able to display a correctly oriented DSS image as long
as a webserver is available to serve the NGS-POSS data. [Patrick
Chevalley of SkyChart](https://github.com/pchev/skychart) has adapted
the NGS-POSS code to Linux and I have written a small web-server for
this purpose around it, which I shall release separately.

`main.py` is the entry-point for the program. Most of the action
happens in `backend.py` which handles various tasks including invoking
the solver and handling some math.

## Alt-Az Workshop Presentation

I presented the general ideas behind this proof-of-concept (along with
a crude demo video) at the 2021 Portland Alt-Az Workshop. The video
recording of the talk is available
[here](https://www.youtube.com/watch?v=VLSbmhEUWO0&t=6855s) and the
slides are available [here](https://asimha.net/AltAzWorkshopTalk2021.pdf)

## White paper

A white paper on the IMU recalibration algorithm is expected
eventually

## Copyright and License

The code (where not sourced from other projects) is Copyright (c)
Akarsh Simha 2023. My code is licensed and released under GNU GPLv3.

