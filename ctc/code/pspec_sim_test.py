#!/usr/bin/env python

"""

NAME: 
      pspec_sim_test.py 
PURPOSE:
      -Tests correlation between maps generated by pspec_sim_v2.py
EXAMPLE CALL:
      ./pspec_sim_test.py --pspec pspec50lmax200 --nchan 50 --sdf 0.001
AUTHOR:
      Carina Cheng

"""

import aipy
import numpy
import pylab
import pyfits
import matplotlib.pyplot as plt
import optparse
import os, sys
import healpy
import scipy
from scipy.interpolate import interp1d
from scipy import integrate

o = optparse.OptionParser()
o.set_usage('fitstopng.py [options]')
o.set_description(__doc__)
o.add_option('--pspec', dest='pspec', default='pspec50lmax200',
             help='Directory where pspec images are contained.')
o.add_option('--nchan', dest='nchan', default=203, type='int',
             help='Number of channels in simulated data. Default is 203.')
o.add_option('--sfreq', dest='sfreq', default=0.1, type='float',
             help='Start frequency (GHz). Default is 0.1.')
o.add_option('--sdf', dest='sdf', default=0.1/203, type='float',
             help='Channel spacing (GHz).  Default is 0.1/203.')
o.add_option('--lmax', dest='lmax', default=5, type='int',
             help='Maximum l value. Default is 5.')
opts, args = o.parse_args(sys.argv[1:])

path = '/Users/carinacheng/capo/ctc/images/pspecs/' + opts.pspec

"""
###plot one pixel value vs. frequency of maps

fluxes = []
freqs = numpy.linspace(opts.sfreq,opts.sfreq+opts.sdf*opts.nchan,num=opts.nchan, endpoint=False) #array of frequencies

for root, dirs, filenames in os.walk(path):
    for f in filenames:
        if f[9:] == '.fits':
            img = aipy.map.Map(fromfits = path + '/' + f)
            flux_px = img.map.map[0:10] #first 10 pixel values
            fluxes.append(flux_px)

plt.plot(freqs, fluxes, 'k-')
plt.xlabel('frequency [GHz]')
plt.ylabel('pixel value')
plt.show()
"""

###get C_l estimators to compare with theoretical values

#cosmology parameters

nu21 = 1420.*10**6 #21cm frequency [Hz]
c = 3.*10**5 #[km/s]
H0 = 100 #69.7 #Hubble's constant [km/s/Mpc]
omg_m = 0.28 
omg_lambda = 0.72

#P_k function

def P_k(kmag, sigma=0.01, k0=0.02):

    return kmag*0+1. #flat P(k)
    #return numpy.exp(-(kmag-k0)**2/(2*sigma**2)) 

#C_l function

def C_l(freq1, freq2, Pk_interp, l_vals): #freqs are entered in GHz
    
    f1 = freq1*10**9 #[Hz]
    f2 = freq2*10**9 #[Hz]
    z1 = (nu21/f1)-1
    z2 = (nu21/f2)-1

    one_over_Ez = lambda zprime:1/numpy.sqrt(omg_m*(1+zprime)**3+omg_lambda)
    Dc1 = (c/H0)*integrate.quad(one_over_Ez,0,z1)[0]
    Dc2 = (c/H0)*integrate.quad(one_over_Ez,0,z2)[0]

    C_ls = []

    for i in range(len(l_vals)):

        ans2=0
        for kk in range(len(k_data)):
            val = (2/numpy.pi)*Pk_interp(k_data[kk])*(k_data[kk]**2)*scipy.special.sph_jn(l_vals[i],k_data[kk]*Dc1)[0][l_vals[i]]*scipy.special.sph_jn(l_vals[i],k_data[kk]*Dc2)[0][l_vals[i]]
            ans2+=val
        ans2*=(k_data[1]-k_data[0])

        C_ls.append(ans2)

    return C_ls

#single freq map C_l theoretical values

#FREQ1 = 0.1 #change these if needed
#FREQ2 = 0.149

k_data = numpy.arange(0.001,10,0.1) #actual data points
Pk_data = P_k(k_data)
Pk_interp = interp1d(k_data, Pk_data, kind='linear')
l_vals = numpy.arange(0,opts.lmax,1) 
"""
#check single freq maps

Cl_th = C_l(FREQ1,FREQ1,Pk_interp,l_vals)

nums = numpy.arange(1,2,1)

avg_cls = numpy.zeros((1,50))

for i in range(len(nums)):

    all_cls = []

    singlemap = aipy.map.Map(fromfits = '/Users/carinacheng/capo/ctc/images/test' + str(nums[i])+ '.fits')#path + '/pspec1001.fits', interp=True)
    alms = singlemap.to_alm(opts.lmax,opts.lmax,iter=1)

    for l in range(opts.lmax):

        prefactor = 1./(2*l+1)
        m = l
        cl = 0

        for j in range(2*l+1):

            if m >= 0:
                alm = numpy.abs((alms[l,m]))**2
                cl += alm
            else:
                alm = numpy.abs((numpy.conj(alms[l,-m])))**2
                cl += alm
            m -= 1

        cl = cl*prefactor
        all_cls.append(cl)

    avg_cls = numpy.add(numpy.array(avg_cls), numpy.array(all_cls))

ans= (numpy.array(avg_cls)/len(nums))[0]
errors = []

for i in range(len(ans)):

    error = numpy.sqrt(2./(len(nums)*(2*l_vals[i]+1)))*Cl_th[i]
    errors.append(error)

p1 ,= plt.plot(l_vals,Cl_th,'k.')
p2 = plt.errorbar(l_vals,ans,yerr=errors,fmt='.')
plt.title('Freq = 0.1GHz')
plt.xlabel('l')
plt.ylabel('C$_{l}$')
plt.legend([p1,p2],['theoretical','observed'])
plt.show()
"""
"""
#check correlated frequency maps

Cl_th = C_l(FREQ1,FREQ2,Pk_interp,l_vals)

nums = numpy.arange(1,21,1) #number of maps made of each freq

avg_cls = numpy.zeros((1,50))

for i in range(len(nums)):

    all_cls = []

    singlemap1 = aipy.map.Map(fromfits = path + '/pspec' + str(nums[i]) + '001.fits', interp=True)
    singlemap2 = aipy.map.Map(fromfits = path + '/pspec' + str(nums[i]) + '002.fits', interp=True)

    alms1 = singlemap1.to_alm(opts.lmax,opts.lmax)
    alms2 = singlemap2.to_alm(opts.lmax,opts.lmax)

    for l in range(opts.lmax):

        prefactor = 1./(2*l+1)
        m = l
        cl = 0

        for j in range(2*l+1):

            if m >= 0:
                alm1 = numpy.conj(alms1[l,m])
                alm2 = alms2[l,m]
                alm = alm1*alm2
                #alm = (alm1*alm2+numpy.conj(alm2)*numpy.conj(alm1))/2
                cl += alm
            else:
                alm1 = alms1[l,-m]
                alm2 = numpy.conj(alms2[l,-m])
                alm = alm1*alm2
                #alm = (alm1*alm2+numpy.conj(alm2)*numpy.conj(alm1))/2
                cl += alm
            m -= 1

        cl = cl*prefactor
        all_cls.append(cl)

    avg_cls = numpy.add(numpy.array(avg_cls), numpy.array(all_cls))

ans = (numpy.array(avg_cls)/len(nums))[0]
#print ans

errors = []

for i in range(len(ans)):

    error = numpy.sqrt(2./(len(nums)*(2*l_vals[i]+1)))*Cl_th[i]
    errors.append(error)

p1 ,= plt.plot(l_vals,Cl_th,'k.')
p2 = plt.errorbar(l_vals,numpy.real(ans),yerr=errors,fmt='.')
plt.title('Freqs = 0.1GHz & 0.149GHz')
plt.xlabel('l')
plt.ylabel('C$_{l}$')
plt.legend([p1,p2],['theoretical','observed'])
plt.show()
"""

###Check P(k) from maps

#box parameters

num_maps = opts.nchan

delta = .05 #step size in real space [Mpc]    #when choosing these, delta*N < Dc_range_in_box
N = 35 #box size [pixels] #MUST BE ODD
L = N*delta #size range in real space [Mpc]

print 'real space volume [Mpc] =', L,'x',L,'x',L
print 'real space resolution [Mpc] =', delta

T_r = numpy.zeros((N,N,N))

#mapping Dcs to freqs

def D_c(freq):
    f = freq*10**9
    z = (nu21/f)-1
    one_over_Ez = lambda zprime:1/numpy.sqrt(omg_m*(1+zprime)**3+omg_lambda)
    Dc = (c/H0)*integrate.quad(one_over_Ez,0,z)[0]
    return Dc

def delta_freq(delta_Dc,z):
    return (delta_Dc*H0*nu21/c)*(numpy.sqrt(omg_m*(1+z)**3+omg_lambda))/(1+z)**2

freqs_range = numpy.arange(0.05,0.25,0.001)
Dcs = []
for i in range(len(freqs_range)):
    Dcs.append(D_c(freqs_range[i]))

freq_interp = interp1d(Dcs[::-1],freqs_range[::-1],kind='linear')

min_freq = 0.15418719
max_freq = 0.17339901
max_Dc = D_c(min_freq)
min_Dc = D_c(max_freq)
Dc_range_in_box = max_Dc-min_Dc
print 'Dc range = ',Dc_range_in_box, ' Mpc'

Dc_center = ((max_Dc-min_Dc)/2.)+min_Dc
f_center = freq_interp(Dc_center)
z_center = (nu21/(f_center*10**9))-1

#read in maps

print 'Reading in maps...'

freqs = numpy.round(numpy.linspace(opts.sfreq,opts.sfreq+opts.sdf*opts.nchan,num=opts.nchan, endpoint=False),decimals=3) #array of frequencies of maps
map_nums = numpy.arange(1,opts.nchan+1,1)
all_maps = []

for f in range(len(map_nums)):

    print '   '+str(map_nums[f])+'/'+str(len(map_nums))
    img = aipy.map.Map(fromfits=path+'/pspec1'+("%03i" % map_nums[f])+'.fits')
    all_maps.append(img)

#loop over box indices

print 'Filling cube...'

#testmap = aipy.map.Map(nside=512)

for i in range(N):
    print '   '+str(i+1)+'/'+str(N)
    for j in range(N):
        for k in range(N):
            #note: origin is in the center of the cube
            px_x = -(N-1)/2 + i #how many pixels to move by from origin
            px_y = -(N-1)/2 + j
            px_z = -(N-1)/2 + k
            delta_x = px_x*delta #physical distance moved from origin
            delta_y = px_y*delta
            delta_z = px_z*delta

            delta_xyz = numpy.sqrt(delta_x**2+delta_y**2+delta_z**2)    

            Dc = numpy.sqrt(Dc_center**2+delta_xyz**2-(2*Dc_center*(-delta_z)))
            #note: when delta_z is negative, looking at bottom half of cube

            delta_f = delta_freq(delta_z,z_center) #[Hz]

            #f = numpy.round(f_center-delta_f*10**-9,decimals=3)
            f = f_center-delta_f*10**-9
            f_string = str(f)[:len(str(opts.sdf))] #interpolating between 2 freqs
            f_lower = float(f_string)
            f_upper = f_lower+opts.sdf

            theta = numpy.arctan2(delta_y,delta_x)
            phi = numpy.arccos((Dc_center+delta_z)/Dc)

            #map_num = int(((f-opts.sfreq)/opts.sdf)+1)
            #img = all_maps[map_num-1]
            #T_r[i][j][k] = img[theta,phi]
            map_num_lower = int(((f_lower-opts.sfreq)/opts.sdf)+1)
            map_num_upper = int(((f_upper-opts.sfreq)/opts.sdf)+1)
            img_lower = all_maps[map_num_lower-1]
            img_upper = all_maps[map_num_upper-1]
            value_lower = img_lower[theta,phi]
            value_upper = img_upper[theta,phi]
            finterp = interp1d([f_lower,f_upper],[value_lower[0],value_upper[0]],kind='linear')
            T_r[i][j][k] = finterp(f)

            """
            #test whether map looks correct
            if k==1: #map 2
                testmap.put((theta,phi),1.0,T_r[i][j][k])
                count+=1
            """
#testmap.to_fits('testmap.fits',clobber=True)
    
#print T_r
 
T_tilde = numpy.fft.fftn(T_r) #[temp*vol]
kx = numpy.fft.fftfreq(N,delta)
kx = kx.copy(); kx.shape = (kx.size,1,1)
ky = kx.copy(); ky.shape = (1,ky.size,1)
kz = kx.copy(); kz.shape = (1,1,kz.size)
k_cube = numpy.sqrt(kx**2+ky**2+kz**2)

#binning

k_real = k_cube*2*numpy.pi
k_sampling = 0.1
k_bins = numpy.arange(0,numpy.max(k_real)+k_sampling,k_sampling)
num_bins = len(k_bins)-1
temp_squared = numpy.zeros(num_bins)
num_in_bins = numpy.zeros(num_bins)

for i in range(N):
    for j in range(N):
        for k in range(N):
            lower=0
            upper=1
            k_val = k_real[i][j][k]
            for b in range(num_bins):
                if k_val >= k_bins[lower] and k_val <= k_bins[upper]:
                    temp_squared[lower] += numpy.abs(T_tilde[i][j][k]*numpy.conj(T_tilde[i][j][k]))#(numpy.abs(T_tilde[i][j][k]))**2
                    num_in_bins[lower] += 1
                lower+=1
                upper+=1

Pk = temp_squared/num_in_bins
print k_bins, Pk

"""   
k_mags = []
index_sum_max = ((len(kx)-1)/2)*3
i=0
j=0
k=0
k_mags.append(float(numpy.sqrt(kx[i]**2+kx[j]**2+kx[k]**2)))
while (i+j+k) != index_sum_max:
    if i==j and i==k:
        i+=1
        j=0
        k=0
        k_mags.append(float(numpy.sqrt(kx[i]**2+kx[j]**2+kx[k]**2)))
    else:
        j+=1
        k_mags.append(float(numpy.sqrt(kx[i]**2+kx[j]**2+kx[k]**2)))
        k+=1
        k_mags.append(float(numpy.sqrt(kx[i]**2+kx[j]**2+kx[k]**2)))

#print k_mags
#print k_cube
#print T_tilde

print 'Sky Map P(k):'
for mag in range(len(k_mags)):
    Ts = []
    for i in range(len(kx)):
        for j in range(len(kx)):
            for k in range(len(kx)):
                if k_cube[i][j][k] == k_mags[mag]:
                    Ts.append(numpy.abs(T_tilde[i][j][k]))
    value = numpy.mean(numpy.array(Ts)**2)
    print 'k=',("%0.5f" % k_mags[mag]),'   P(k)=', value
    plt.plot(k_mags[mag]*2*numpy.pi,value,'b.')
"""

print 'Sky Map P(k):'
for i in range(len(k_bins)-1):
    print k_bins[i],'<k<',k_bins[i+1],'   P(k)=',Pk[i]
    #plt.plot(k_bins[i+1],P_k[i],'b.')
    y_err = numpy.sqrt(2/num_in_bins[i])*P_k(k_bins[i]+((k_bins[i+1]-k_bins[i])/2))
    plt.errorbar(k_bins[i+1],Pk[i]*33,xerr=1/L,yerr=y_err,fmt='.')

print 'Original P(k):'
for i in range(len(k_data)):
    print 'k=',("%0.5f" % k_data[i]),'   P(k)=', Pk_data[i]
    #plt.plot(k_data[i],Pk_data[i],'k.')
plt.show()

    

