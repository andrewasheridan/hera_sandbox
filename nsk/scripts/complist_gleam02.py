"""
complist_gleam02.py

script for making gleam 020012 -305327
model component list in CASA
"""
import os
import shutil
import numpy as np
import argparse
import sys

args = argparse.ArgumentParser(description="Run with casa as: casa -c complist_gleam02.py <args>")
args.add_argument("-c", type=str, help="name of this script")
args.add_argument("--image", default=False, action='store_true', help='make FITS image of model')
args.add_argument("--freqs", default=None, type=str, help="comma-separated values [MHz] for input into np.linspace({},{},{})")
args.add_argument("--cell", default='200arcsec', type=str, help="image pixel size in arcsec")
args.add_argument("--imsize", default=512, type=int, help="number of pixels in image")

if __name__ == "__main__":
    a = args.parse_args()

    # set variables
    direction = "J2000 02h00m12.7s -30d53m27s"
    ref_freq = "151MHz"

    # gleam 020012 -305327
    cl.addcomponent(label="GLEAM0200-3053", flux=17.5, fluxunit="Jy", dir=direction, freq=ref_freq,
                    shape="point", spectrumtype='spectral index', index=-0.86)

    # gleam 0150 -2931
    cl.addcomponent(label="GLEAM0150-2931", flux=16.6, fluxunit="Jy", dir="J2000 01h50m36s -29d31m59s", freq=ref_freq,
                    shape="point", spectrumtype='spectral index', index=-0.80)

    # gleam 015200 -294056
    cl.addcomponent(label="GLEAM0152-2940", flux=4.7, fluxunit="Jy", dir="J2000 01h52m00s -29d40m56s", freq=ref_freq,
                    shape="point", spectrumtype='spectral index', index=-0.73)

    # gleam 013411 -362913
    cl.addcomponent(label="GLEAM0134-3629", flux=18.5, fluxunit="Jy", dir="J2000 01h34m11.6s -36d29m13s", freq=ref_freq,
                    shape="point", spectrumtype='spectral index', index=-0.72)

    # save
    if os.path.exists("gleam02.cl"):
        shutil.rmtree("gleam02.cl")
    cl.rename("gleam02.cl")

    # make image
    if a.image:
        # get frequencies
        if a.freqs is None:
            Nfreqs = 1
            freqs = np.array([151.0])
        else:
            freqs = np.linspace(*np.array(a.freqs.split(',')).astype(np.float), endpoint=False)
            Nfreqs = len(freqs)

        # setup image
        ia.fromshape("gleam02.cl.im", [a.imsize, a.imsize, 1, Nfreqs], overwrite=True)
        cs = ia.coordsys()
        cs.setunits(['rad','rad','','Hz'])

        # set pixel properties
        cell_rad = qa.convert(qa.quantity(a.cell),"rad")['value']
        cs.setincrement([-cell_rad, cell_rad], type='direction')
        cs.setreferencevalue([qa.convert("02h00m12.7s",'rad')['value'], qa.convert("-30d53m27s",'rad')['value']], type="direction")

        # set freq properties
        qa_freqs = qa.quantity(freqs, 'MHz')
        cs.setspectral(frequencies=qa_freqs)
 
        # set flux properties, make image, export to fits
        ia.setcoordsys(cs.torecord())
        ia.setbrightnessunit("Jy/pixel")
        ia.modify(cl.torecord(), subtract=False)
        print("...saving gleam02.cl.fits")
        exportfits(imagename="gleam02.cl.im", fitsimage="gleam02.cl.fits", overwrite=True, stokeslast=False)

    cl.close()





