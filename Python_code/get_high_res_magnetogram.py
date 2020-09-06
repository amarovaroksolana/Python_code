from astropy.io import fits
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import math

path='C:/Users/amaro/Desktop/Roksolanai/'
file = path + 'mdi.fd_M_96m_lev182.19990514_000000_TAI.data.fits'

hdr = fits.open(file)
#data_B = hdr[1].data
data_B = fits.getdata(file)


x0_B = hdr[1].header['CRPIX1']
y0_B = hdr[1].header['CRPIX2']
dx_B = hdr[1].header['CDELT1']
dy_B = hdr[1].header['CDELT2']
alpha_max = hdr[1].header['RSUN_OBS']

######################################################

xcoords_B = (np.arange(0,1024) - x0_B + 1.0 ) * dx_B
ycoords_B = (np.arange(0,1024) - y0_B + 1.0 ) * dy_B

######################################################
arcsec_to_rad = ( 1.0 / ( 60.0 * 60.0 ) ) * ( math.pi / 180 )

delta = math.sin( alpha_max * arcsec_to_rad )

path = 'C:/Users/amaro/Desktop/Python/'

with open(path+"/19990513/MDI_0000__PFSS_0004/coord_grids.sav", 'rb') as f:
    X_coord_array = np.load(f)
    Y_coord_array = np.load(f)
    Z_coord_array = np.load(f)

    
x0 = X_coord_array[20, 20]
y0 = Y_coord_array[ 20, 20]
z0 = Z_coord_array[ 20, 20]

######################################################
eta0 = math.atan( x0 / z0 )

a_norm_x = math.cos( eta0 )
a_norm_y = 0.0
a_norm_z = -math.sin( eta0 )

rho0 = math.sqrt( ( z0 ** 2.0 ) + ( x0 ** 2.0 ) )

ksi0 = math.acos( rho0 )

b_norm_x = -math.sin( ksi0 ) * math.sin( eta0 )
b_norm_y = rho0
b_norm_z = -math.sin( ksi0 ) * math.cos( eta0 )

######################################################
x_rel_array = np.zeros(( 161, 161 ))
y_rel_array = np.zeros(( 161, 161 ))
b_radial_array = np.zeros(( 161, 161 ))
######################################################


a1 = 675
a2 = 715
b1 = 686
b2 = 726
b_radial_array = []
x_rel_array = []
y_rel_array = []
for a in range (a1-60, a2+61):
    for b in range (b1-60, b2+61):
        
######################################################
        b_los = data_B[b,a]

        a_coord = xcoords_B[a]
        b_coord = ycoords_B[b]
        
        a_rad = a_coord * arcsec_to_rad
        b_rad = b_coord * arcsec_to_rad
       
        tan_a = math.tan( a_rad )
        tan_b = math.tan( b_rad )
        cos_a = math.cos( a_rad )
    
        gamma = ( tan_a ** 2.0 ) + ( ( tan_b * cos_a ) ** 2.0 )
     
        z = ( ( math.sqrt( ( delta ** 2.0 ) - ( gamma * ( 1 - ( delta ** 2.0 ) ) ) ) + gamma ) / ( 1 + gamma ) ) / delta
        x = ( ( 1 / delta ) - z ) * tan_a
        y = ( ( 1 / delta ) - z ) * tan_b * cos_a
    
        delta_x = x - x0
        delta_y = y - y0
        delta_z = z - z0
    
        x_rel = ( delta_x * a_norm_x ) + ( delta_y * a_norm_y ) + ( delta_z * a_norm_z )
        y_rel = ( delta_x * b_norm_x ) + ( delta_y * b_norm_y ) + ( delta_z * b_norm_z )
    
        b_r = b_los / z

        b_radial_array.insert( (b-(b1-60))*161 + (a-(a1-60)), b_r ) 
        x_rel_array.insert( (b-(b1-60))*161 + (a-(a1-60)), x_rel ) 
        y_rel_array.insert( (b-(b1-60))*161 + (a-(a1-60)), y_rel ) 

######################################################
        
x_interpolated = ( np.arange(321.)*0.001 ) - 0.16
y_interpolated = ( np.arange(321.)*0.001 ) - 0.16

############################################################################################################

plt.imshow((data_B[ b1:b2, a1:a2]), origin = 'lower', cmap = 'gray', extent = [0,40, 0, 40])
plt.show()

############################################################################################################

xi,yi = np.meshgrid(x_interpolated,y_interpolated)

b_radial_interpolated = griddata((x_rel_array, y_rel_array), b_radial_array, (xi, yi), method='linear')
#b_radial_interpolated = b_radial_interpolated.flatten()
plt.imshow(b_radial_interpolated, extent=(-0.15,0.15,-0.15,0.15), origin='lower', cmap = 'gray')

plt.title('Linear')
plt.show()

############################################################################################################

"""
import idlmagic
from idlpy import *
IDL.x = x_rel_array
IDL.y = y_rel_array

IDL.run('Triangulate, x, y, triangles')
tri = IDL.triangles
"""
