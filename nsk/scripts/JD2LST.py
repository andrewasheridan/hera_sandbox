#!/usr/bin/env python2.7

from astropy.time import Time
import argparse

a = argparse.ArgumentParser()
a.add_argument('--jd', default=[], type=float, help='JD float to convert to LST')
a.add_argument('--lon', default=[], type=float, help='longitude float of observer in degrees East')

def JD2LST(JD, longitude):
    """
    Input:
    JD : float, julian date
    longitude : float, longitude in degrees East

    Output:
    Local Apparent Sidreal Time in Hour Angle
    """
    t = Time(JD, format='jd', scale='utc')
    return t.sidereal_time('apparent', longitude=longitude).value


def LST2JD(LST, start_JD, longitude):
    """
    calculate LST -> JD quickly via a linear fit

    Input:
    ------
    LST : local sidereal time, hour angle
    start_JD : staring JD to anchor LST2JD conversion
    longitude : degrees east of observer

    Output:
    JD : float of JD, accurate to ~1 milliseconds
    """
    base_JD = float(start_JD)
    while True:
        # calculate fit
        jd1 = start_JD
        jd2 = start_JD + 0.01
        lst1, lst2 = JD2LST(jd1, longitude=longitude), JD2LST(jd2, longitude=longitude)
        slope = (lst2 - lst1) / 0.01
        offset = lst1 - slope * jd1

        # solve y = mx + b for x
        JD = (LST - offset) / slope

        # redo if JD isn't on starting JD
        if JD - base_JD < 0:
            start_JD += 1
        elif JD - base_JD > 1:
            start_JD -= 1
        else:
            break

    return JD


if __name__ == "__main__":
    args = a.parse_args()
    if args.jd is not None and args.lon is not None:
        print JD2LST(args.jd, args.lon)
