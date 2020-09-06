from astropy.io import fits
import numpy as np
import math

path='C:/Users/amaro/Desktop/Roksolanai/'
file = path + 'mdi.fd_M_96m_lev182.19990514_000000_TAI.data.fits'

hdr = fits.open(file)

x0_B = hdr[1].header['CRPIX1']
y0_B = hdr[1].header['CRPIX2']
dx_B = hdr[1].header['CDELT1']
dy_B = hdr[1].header['CDELT2']
alpha_max = hdr[1].header['RSUN_OBS']

######################################################

xcoords_B = (np.arange(0,1024) - x0_B + 1.0 ) * dx_B
ycoords_B = (np.arange(0,1024) - y0_B + 1.0 ) * dy_B

######################################################

X_coord_array = np.zeros(( 41, 41 ), float)
Y_coord_array = np.zeros(( 41, 41 ), float)
Z_coord_array = np.zeros(( 41, 41 ), float)

arcsec_to_rad = ( 1.0 / ( 60.0 * 60.0 ) ) * ( math.pi / 180 )

delta = math.sin( alpha_max * arcsec_to_rad )

a1 = 675
a2 = 715
b1 = 686
b2 = 726

#a1 = 675
#a2 = 716
#b1 = 686
#b2 = 727
######################################################

for a in range (a1, a2+1):
    for b in range (b1, b2+1):
        a_coord = xcoords_B[a]
        b_coord = ycoords_B [b]

        a_rad = a_coord * arcsec_to_rad
        b_rad = b_coord * arcsec_to_rad

        tan_a = math.tan( a_rad )
        tan_b = math.tan( b_rad )
        cos_a = math.cos( a_rad )
        
        gamma = ( tan_a ** 2.0 ) + ( ( tan_b * cos_a ) ** 2.0 )
        
        z = ( ( math.sqrt( ( delta ** 2.0 ) - ( gamma * ( 1 - ( delta ** 2.0 ) ) ) ) + gamma ) / ( 1 + gamma ) ) / delta
        x = ( ( 1 / delta ) - z ) * tan_a
        y = ( ( 1 / delta ) - z ) * tan_b * cos_a

        X_coord_array [a-a1, b-b1] = x
        Y_coord_array [a-a1, b-b1] = y
        Z_coord_array [a-a1, b-b1] = z

a_coords = xcoords_B [a1 : a2+1]
b_coords = ycoords_B [b1 : b2+1]     


path = 'C:/Users/amaro/Desktop/Python/'

with open(path+"/19990513/MDI_0000__PFSS_0004/coord_grids.sav", 'wb') as f:  # Python 3: open(..., 'wb')
    np.save(f, X_coord_array)
    np.save(f, Y_coord_array)
    np.save(f, Z_coord_array)
    np.save(f, a_coords)
    np.save(f, b_coords)

