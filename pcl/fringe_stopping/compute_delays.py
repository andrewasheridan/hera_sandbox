# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from astropy import constants, coordinates, units
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, FK5

from pyuvdata import utils as uvutils
from hera_mc import geo_handling

# get cofa information
handling = geo_handling.Handling()
cofa = handling.cofa()[0]
cofa_lat = cofa.lat * np.pi / 180.0
cofa_lon = cofa.lon * np.pi / 180.0
cofa_alt = cofa.elevation

cofa_xyz = uvutils.XYZ_from_LatLonAlt(cofa_lat, cofa_lon, cofa_alt)
cofa_enu = uvutils.ENU_from_ECEF(cofa_xyz, cofa_lat, cofa_lon, cofa_alt)

# get antennas
telescope_location = EarthLocation.from_geocentric(*cofa_xyz, unit=units.m)
jd = int(Time.now().jd)
if True:
    time0 = Time.now()
    time0.location = telescope_location
else:
    time0 = Time(jd, format="jd", location=telescope_location)
print("time0: ", time0.jd)
ants = handling.get_active_stations(time0, "all")
# sort the list by antenna name
ants = sorted(ants, key=lambda x: int(x.station_name[2:]))

# convert from lat/lon/alt to xyz
N_ants = len(ants)
ant_lat = np.zeros(N_ants, dtype=np.float64)
ant_lon = np.zeros(N_ants, dtype=np.float64)
ant_alt = np.zeros(N_ants, dtype=np.float64)
ant_names = []

for i, ant in enumerate(ants):
    ant_lat[i] = ant.lat * np.pi / 180.0
    ant_lon[i] = ant.lon * np.pi / 180.0
    ant_alt[i] = ant.elevation
    ant_names.append(ant.station_name)

antpos_xyz = uvutils.XYZ_from_LatLonAlt(ant_lat, ant_lon, ant_alt)

# Compute time samples to use for delays.
# Our sampling rate is (250e6)/8192 spectra/s.
# We want our integration to be 2**19 spectra, which gives t_int ~ 17.2s.
# Within this window, we will use 512 different phases, setting our fringe stopping time
N_times = 512
samp_rate = 250e6 / 8192
t_int = 2 ** 19 / samp_rate
t_fs = t_int / N_times

# lay out grid of times to use
t_center = t_int / 2.0
times = np.zeros(N_times, dtype=np.float64)
for i in range(N_times):
    times[i] = t_center - t_int + (i + 0.5) * t_fs

# make to astropy time objects and define phase center (RA of zenith at center
# of observation)
times_jd = time0 + times * units.s
lst_radians = time0.sidereal_time("apparent").to("rad").value
itrs_telescope_locations = telescope_location.get_itrs(obstime=times_jd)
phase_center_coord = SkyCoord(
    ra=lst_radians, dec=cofa_lat, unit="radian", equinox=time0, frame=FK5
)
icrs_coord = phase_center_coord.transform_to("icrs")
frame_phase_center = icrs_coord

frame_telescope_locations = itrs_telescope_locations.transform_to(coordinates.ICRS)
frame_telescope_locations.representation_type = "cartesian"

# make empty array of phases
delay_values = np.zeros((N_ants, N_times))

for i in range(N_times):
    time = times_jd[i]
    frame_telescope_location = frame_telescope_locations[i]
    itrs_ant_coord = SkyCoord(
        x=antpos_xyz[:, 0] * units.m,
        y=antpos_xyz[:, 1] * units.m,
        z=antpos_xyz[:, 2] * units.m,
        frame="itrs",
        obstime=time,
    )
    frame_ant_coord = itrs_ant_coord.transform_to("icrs")
    frame_ant_rel = (
        (frame_ant_coord.cartesian - frame_telescope_location.cartesian)
        .get_xyz()
        .T.value
    )
    frame_ant_uvw = uvutils.phase_uvw(
        frame_phase_center.ra.rad, frame_phase_center.dec.rad, frame_ant_rel
    )
    delay_values[:, i] = frame_ant_uvw[:, 2] / constants.c.to("m/s").value

dtaus = delay_values[:, -1] - delay_values[:, 0]
assert len(dtaus) == N_ants
print("max delay: ", np.amax(np.abs(dtaus)))

# convert to a dictionary
channel_width = 250e6 / 8192
delay_dict = {}
for i, ant_name in enumerate(ant_names):
    # convert to form required by FPGA block
    fringe_phase = 2 * np.pi * delay_values[i, :] * channel_width * (180.0 / np.pi)
    # make sure angle is between -pi and pi
    np.where(fringe_phase > 180, fringe_phase - 360, fringe_phase)
    np.where(fringe_phase < -180, fringe_phase + 360, fringe_phase)
    delay_dict[ant_name] = fringe_phase.tolist()

# dump using json
with open("bda_phases.json", "w") as fp:
    json.dump(delay_dict, fp)
