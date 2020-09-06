from astropy.io import fits
import numpy as np
import math
import h5py
import pfsspy
import scipy
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from functions import get_interpolation_index, gaussquad_legendre, spherical_transform, pfss_get_potl_coeffs,pfss_potl_field__AV 

path='C:/Users/amaro/Desktop/Roksolanai/'
file = path + 'mdi.fd_M_96m_lev182.19990514_000000_TAI.data.fits'

R_Sun = 695700000.0 #in [m] assuming Sun radius of 695700 [km]

hdr = fits.open(file)
#data_B = hdr[1].data
data_B = fits.getdata(file)

#read fits file
x0_B = hdr[1].header['CRPIX1']
y0_B = hdr[1].header['CRPIX2']
dx_B = hdr[1].header['CDELT1']
dy_B = hdr[1].header['CDELT2']
alpha_max = hdr[1].header['RSUN_OBS']
B_0 = hdr[1].header['CRLT_OBS']
L_0 = hdr[1].header['CRLN_OBS']


xcoords_B = (np.arange(0,1024) - x0_B + 1.0 ) * dx_B
ycoords_B = (np.arange(0,1024) - y0_B + 1.0 ) * dy_B

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

eta0 = math.atan( x0 / z0 )

rho0 = math.sqrt( ( z0 ** 2.0 ) + ( x0 ** 2.0 ) )

ksi0 = math.acos( rho0 )

a1 = 675
a2 = 715
b1 = 686
b2 = 726

a10 = 635
a20 = 795
b10 = 646

no_high_res_pixels = a20 - a10 + 1
b20 = b10 + ( a20 - a10 )

B_0_rad = ( B_0 / 180.0 ) * math.pi
L_0_rad = ( L_0 / 180.0 ) * math.pi
sin_B_0 = math.sin( B_0_rad )
cos_B_0 = math.cos( B_0_rad )

###################################################################
path='C:/Users/amaro/Desktop/Roksolanai/'
with h5py.File(path+"Bfield_19990514_000400.h5", 'r') as f:
    lat_global = f['ssw_pfss_extrapolation'][0][4]
    lon_global = f['ssw_pfss_extrapolation'][0][5]
    br = f['ssw_pfss_extrapolation'][0][8]

mag_global = br[0].T
#mag_global = []
#for i in range (73728):
    #mag_global.append(br[i])


nlat_max=1937  #number of latitudinal gridpoints in magnetogram
nlat = nlat_max
nlon=nlat*2

cth = gaussquad_legendre(nlat)
theta = np.arccos(cth[0])

lat=90-theta*180/math.pi


lon=np.linspace(0,360, nlon+1)[0:nlon]
phi=lon*(math.pi/180)


dlatinterp=get_interpolation_index(lat_global, lat)
dloninterp=get_interpolation_index(lon_global, lon)
#mags_interpolated = scipy.ndimage.map_coordinates(mag_global, [dlatinterp, dloninterp], order=1, mode='nearest')

#xi,yi =np.meshgrid(dlatinterp,dloninterp)

###################################################################

#mags_interpolated = scipy.interpolate.LinearNDInterpolator( mag_global, (dloninterp, dlatinterp))
#mags_interpolated = scipy.interpolate.RectBivariateSpline(xi, yi,mag_global)
#dlatinterp = np.all(np.diff(dlatinterp) > 0)
#dloninterp = np.all(np.diff(dloninterp) > 0)

#mags_interpolated = scipy.interpolate.RectBivariateSpline(dlatinterp, dloninterp, mag_global, kind='linear')
#mags_interpolated = scipy.interpolate.bisplev(dlatinterp, dloninterp, mag_global)
#xi,yi = np.meshgrid(x_interpolated,y_interpolated)

#mags_interpolated = griddata((dlatinterp,dloninterp), mag_global, (xi, yi), method='linear')
#mags_interpolated = scipy.ndimage.interpolation.map_coordinates((dlatinterp,dloninterp), mag_global)
#mags_interpolated = bilinear_interpolation( dlatinterp,dloninterp, mag_global.flatten())

###################################################################

a_high_res_array = np.zeros( (no_high_res_pixels, no_high_res_pixels ))
b_high_res_array = np.zeros(( no_high_res_pixels, no_high_res_pixels ))
b_radial_array = np.zeros(( no_high_res_pixels,no_high_res_pixels ))

###################################################################
no_pixel = 0
a_high_res_array = [] 
b_high_res_array = []
b_radial_array = []

for a in range (a10, a20+1):
    for b in range (b10, b20+1):
        
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
        x = ( ( 1.0 / delta ) - z ) * tan_a
        y = ( ( 1.0 / delta ) - z ) * tan_b * cos_a
    
        x_prim = x 
        y_prim = ( y * cos_B_0 ) + ( z * sin_B_0 )
        z_prim = ( y * (-sin_B_0 ) ) + ( z * cos_B_0 )

        #Here calculate the spherical coordinates of the volume element in the Sun's reference frame.
        r_coord = math.sqrt( ( x_prim**2 ) + ( y_prim**2 ) + ( z_prim**2 ) )

        if ( r_coord - 1.0 ) > 0.0001:
            print ('a = ', a, '; b = ', b, '; diff = ', r_coord - 1.0)

        rho_0_coord = math.sqrt( ( x_prim**2 ) + ( z_prim**2 ) )

        cos_theta_coord = y_prim / r_coord
        sin_theta_coord = rho_0_coord / r_coord
        cos_phi_coord = z_prim / rho_0_coord
        sin_phi_coord = x_prim / rho_0_coord

        theta_coord = math.acos( cos_theta_coord )

  
        if x_prim >= 0:
            phi_coord = math.acos( cos_phi_coord )
        else:
            phi_coord = ( 2 * math.pi ) - math.acos( cos_phi_coord )

        phi_coord = L_0_rad + phi_coord

        if phi_coord > ( 2 * math.pi ):
            phi_coord = phi_coord - ( 2 * math.pi )

        b_r = b_los / z

        a_high_res_array.insert( no_pixel, (phi_coord / math.pi ) * 180.0 ) 
        b_high_res_array.insert( no_pixel, ( ( theta_coord / math.pi ) * 180.0 * (-1.0) ) + 90.0)
        b_radial_array.insert( no_pixel, b_r)
 
        no_pixel = no_pixel + 1


#########################################################################################
x_interpolated = lon[2260 : 2464+1]
y_interpolated = lat[1117 : 1340+1]
xi,yi = np.meshgrid(x_interpolated,y_interpolated)

b_high_res_interpolated = griddata((a_high_res_array, b_high_res_array), b_radial_array, (xi, yi), method='linear')

b_high_res_interpolated = b_high_res_interpolated.flatten()

##########################################################################################

#mags_improved = mags_interpolated
#mags_improved[226 : 2464][1117 : 1340 ] = b_high_res_interpolated

#next get PFSS coefficients
rss=1.6  #source surface radius
#pfss_get_potl_coeffs( mags_improved, rss, None)

phibt [0, 0] = complex(0, 0)

#get l and m index arrays of transform
phisiz=np.shape(phiat)
#A.V.: returns the size of each of dimensions in phiat
lix=np.arange(0,phisiz[0])
#A.V.: creates an array with values corresponding to indices: 0, 1, 2, ..., phisiz(0)-1
mix=np.arange(0,phisiz[1])
larr=lix*(np.repeat(1,phisiz[1]))[:, np.newaxis]
#A.V.: replicate here will make an array the size of phisiz(1) with value 1 for each of the elements
#A.V.: # multiplies the two arrays
marr=np.repeat(1,phisiz[0])*mix[:, np.newaxis]

wh=np.nonzero(marr > larr)
larr[wh]==0  &  marr[wh]==0

max_z = 0.40 # max height expressed in Sun radius
#d_r_coord_m = 100000.0 ; 100 [km] in [m]
#d_r_coord = d_r_coord_m / R_Sun
d_r_coord = 0.00016 # in [R_Sun]
N_r_pfss = round( max_z / d_r_coord ) + 1
#N_r_pfss = 3
r_coords = 1 + ( np.arange(0,N_r_pfss ) * d_r_coord )

phi_1 = 3.62 # in [rad]
#phi_2 = 3.91 # in [rad]
N_phi = 180.0
#d_phi_rad = ( phi_2 - phi_1 ) / ( N_phi - 1 )
d_phi_rad = 0.00162 # in [rad]
phi_coords_pfss = ( np.arange(0,N_phi ) * d_phi_rad ) + phi_1

theta_1 = 1.35 # in [rad]
#theta_2 = 1.145 # in [rad]
N_theta = 174.0
#d_theta_rad = ( theta_1 - theta_2 ) / ( N_theta - 1 )
d_theta_rad = 0.00162 # in [rad]
theta_coords_pfss = -( np.arange( 0, N_theta ) * d_theta_rad ) + theta_1

#for z_index = 0, N_r_pfss do begin
for z_index in range (0,2):

    l_limit = nlat_max
    # next reconstruct the coronal field in a spherical shell between 1 and rss
    """pfss_potl_field__AV(max_z, 3, r_coords(z_index), theta_coords_pfss, phi_coords_pfss, l_limit, None, None, None)
    ;pfss_potl_field, max_z, 3, rindex=r_coords, thindex=theta_coords_pfss, phindex=phi_coords_pfss, lmax = nlat0, /trunc
    ;usage: pfss_potl_field,rtop,rgrid,rindex=rindex,thindex=thindex,
    ;           phindex=phindex,lmax=lmax,/trunc,potl=potl,/quiet
    ;         where rtop=radius of uppermost gridpoint
    ;               rgrid=sets radial gridpoint spacing:
    ;                      1 = equally spaced (default)
    ;                      2 = grid spacing varies with r^2
    ;                      3 = custom radial grid given by the rindex keyword
    ;               rindex = custom array of radial coordinates for output grid
    ;               thindex = (optional) custom array of theta (colatitude)
    ;                         coordinates, in radians, for output grid.  If not
    ;                         specified existing latitudinal grid is used.
    ;               phindex = (optional) custom array of phi (longitude)
    ;                         coordinates, in radians, for output grid.  If not
    ;                         specified, existing longitudinal grid is used.
    ;               lmax=if set, only use this number of spherical harmonics in
    ;                    constructing the potential (and thus the field)
    ;               trunc=set to use fewer spherical harmonics when
    ;                     reconstructing B as you get farther out in radius
    ;               potl=contains potl if desired, but what you pass
    ;                    to this routine must not be undefined in order
    ;                    for the field potential to be computed
    ;               quiet = set for minimal screen output

    ;pfss_to_spherical,pfss_sph_data"""
    
    z_subtext = str(np.fix( z_index ))

    if z_index > 10:
        z_text = '000' + z_subtext
    else:
        if z_index > 100:
            z_text = '00' + z_subtext
        else:
            if z_index > 1000:
                z_text = '0' + z_subtext
            else:
                z_text = z_subtext

  
    #path = 'C:\Users\amaro\Desktop\Roksolanai\19990513\MDI_0000__PFSS_0004\RSS_1_6\Fields\'
    #path = 'C:\Users\artur\NextCloud\Zinatne\Saules_fizika\Plankumu_izpetes_algoritms\19990513\MDI_0000__PFSS_0004\RSS_1_6\Fields\'
    #path = 'C:\Users\Arturs\NextCloud\Zinatne\Saules_fizika\Plankumu_izpetes_algoritms\19990513\MDI_0000__PFSS_0004\RSS_1_6\Fields\'
    #save, br, bph, bth, filename=path+'fields_layer_'+z_text+'.sav'
    #save, br, bph, bth, filename=path+'fields_layer_0001.sav'

    print('Layer ', z_text, ' saved.')
    

path='C:/Users/amaro/Desktop/Python/'
with open(path+"/19990513/MDI_0000__PFSS_0004/pfss_data_block.sav", 'wb') as f:  
    np.save(f, theta)
    np.save(f, nlat)
