import numpy as np
import time 
path='C:/Users/amaro/Desktop/Python/'
with open(path+"/19990513/MDI_0000__PFSS_0004/pfss_data_block.sav", 'rb') as f:
    theta = np.load(f)
    nlat = np.load(f)
    #phiat = np.load(f)
    #phibt = np.load(f)

###CORRECT!!!
def gaussquad_legendre (n):
    order = n
    if order < 2:
        print("gaussquad_legendre: n must be at least 2")
        return
    
    #setup
    x=np.zeros(order)
    pprime=np.zeros(order)

    #set tolerance
    eps=9.9999999999999995e-007 #(1d-6)probably adequate for real*8 double precision

    #loop through points
    for i in range (1,int((order+1)/2)+1) :  #symmetric domain, only do half of the points
        #starting guess for ith root
        guess=np.cos(np.pi*(i-np.float64(0.25e0))/(order+np.float64(0.5e0)))

        #iterate until zero is found
        while True:
            #  starting values for P_(n-1) and P_n, evaluated at guess point
            pnm1 = np.float64(1.0e0)
            pn = guess
        
            #find P_n evaluated at guess point, Arfken (12.17a) after n+1 replaces n
            for j in range (2,order+1):
                pnm2 = pnm1
                pnm1 = pn
                pn = (guess*(2*j-1)*pnm1-(j-1)*pnm2)/j

            #compute d P_n / dx evaluated at guess point, basically Arfken (12.26)
            dpndx = order*(guess*pn-pnm1)/(guess*guess-1)

            #use Newton's method to improve guess
            oldguess = guess
            guess = oldguess-pn/dpndx

            #check tolerance, repeat if answer isn't good enough

            if abs(guess-oldguess)<= eps : break
        #fill x and pprime arrays
        x[i-1]=-guess
        x[order-i]=guess
        pprime[i-1]=dpndx  #may be off by a minus sign, but it gets squared below
        pprime[order-i]=dpndx  #ditto above


    #calculate weights
    w=2/((1-x*x)*pprime*pprime)
    
    return x,w

###CORRECT!!!
def  get_interpolation_index (array,value):
    
    npt= len(value)
    out=np.zeros(npt)
    for i in range (0,npt):
        if value[i] <= array[0]:
            out [i] = np.float64(0e0)
        else:
            nearix = np.nonzero(array > value[i])
            nwh = np.count_nonzero(nearix)
            if ( nwh > 0):
               nix=nearix[0]-1
               extract=array[nix[0]:nix[0]+2]
               out[i]=nix[0]+(value[i]-extract[0])/float(extract[1]-extract[0])
            else:
               out[i]=np.double(len(array)-1)
                
    return  out

###CORRECT!!!
def weights_legendre(x):
    #preliminaries
    nx=len(x)
    weights=np.zeros(nx)
    costheta=x
    sintheta=np.sqrt(1-(costheta**2))
    #set first two Legendre functions (evaluated at the collocation points)
    Pm2 = 1
    Pm1 = x
    
    #iterate through the rest of the functions
    for l in range (2, nx):
        lr=1/l
        P=(2-lr)*Pm1*costheta-(1-lr)*Pm2  #recursion relation Arfken 12.17a
        Pm2=Pm1
        Pm1=P
        
    p_deriv=(nx*P)/sintheta**2  #recursion relation Arfken 12.26

    #calculate and then renormalize the weights
    weights=2/(sintheta*p_deriv)**2
    weights = weights * (2 * np.pi)

    return weights
################################################################
#Check spherical_transform
def spherical_transform (A,cp,lmax,period):
    costheta=np.float64(cp)
    sintheta=np.sqrt(1-costheta**2)

    sz = np.shape(A)
    n_phi=sz[0]
    n_theta=sz[1]
    
    if lmax is None:
        lmax=(2*n_theta-1)/3
    else:
        lmax=np.long(lmax)
    if period is None:
        period=1
    else:
        period=np.fix(period[0])
        
    weights=weights_legendre(costheta)
    Bm = np.zeros((n_phi, n_theta), dtype = complex)
    #Bm=np.array(Bm,dtype = complex)
    
    for i in range (0,n_theta):
        Bm[:][i]=np.fft.fft((A[:][i]))
        
    #finally the Legendre transform: theta -> l
    B=np.zeros((lmax+1,int(lmax/period)+1), dtype = complex)  #only half of this array will be filled
    #B=np.array(B,dtype = complex)
    
    #Define N_mm such that Y_mm = N_mm sin^m(theta) exp(i m phi) i.e. it's the
    #normalization for the sectoral harmonics.  It will be useful below in
    #computing the spherical harmonics recursively.
    N_mm=np.zeros(lmax+1)
    N_mm[0]=1/np.sqrt(np.float64(4e0)*np.pi)
    
    for m in range (1,lmax+1):
        N_mm[m]=-N_mm[m-1]*np.sqrt(1+1/np.float64(2*m))

    #first do m=0
    P_lm2=N_mm[0]
    P_lm1=P_lm2*costheta*np.sqrt(np.float64(3e0))
   
    B[0][0]=np.sum(Bm[0][:]*P_lm2*weights)  #set l=0 m=0 term
    B[1][0]=np.sum(Bm[0][:]*P_lm1*weights)  #set l=1 m=0 term
    
    for l in range (2,lmax+1) :
        lr=np.double(l)
        c1=np.sqrt(np.float64(4e0)-1/lr**2)
        c2=-(1-1/lr)*np.sqrt((2*lr+1)/(2*lr-3))
        P_l=c1*costheta*P_lm1+c2*P_lm2
        B[l][0]=np.sum(Bm[0][:]*P_l*weights)  #set m=0 term for all other l's
        P_lm2=P_lm1
        P_lm1=P_l

    #note factor of 2 below accounts for the way IDL distributes power
    #in its fft, since only the l modes from 1 to lmax are used below
    Bm=2*Bm

    #now the rest of the m's
    old_Pmm=N_mm[0]
    
    for m in range (1,lmax+1):
        P_lm2 = old_Pmm*sintheta*N_mm[m]/N_mm[m-1]
        P_lm1 = P_lm2*costheta*np.sqrt(np.double(2*m+3))
        old_Pmm = P_lm2
        if (m % period) == 0:
            B[m][int(m/period)]=np.sum(Bm[int(m/period)][:]*P_lm2*weights)  #set l=m term
            if m < lmax:
                B[m+1][int(m/period)] = np.sum(Bm[int(m/period)][:]*P_lm1*weights)  #set l=m+1 term
            mr=np.double(m)
            for l in range (m+2,lmax+1):
                lr=np.double(l)
                c1=np.sqrt((4*lr**2-1)/(lr**2-mr**2))
                c2=-np.sqrt(((2*lr+1)*((lr-1)**2-mr**2))/((2*lr-3)*(lr**2-mr**2)))
                P_l=c1*costheta*P_lm1+c2*P_lm2

                B[l][int(m/period)]=np.sum(Bm[int(m/period)][:]*P_l*weights)
                #old version of prev statement is below, corrected 14 Sep 2000
                #B(l,m/period)=N_mm(m)*total(Bm(m/period,*)*P_l*weights,/double)

                P_lm2=P_lm1
                P_lm1=P_l
                
    return B
    
###############################################################################
#Check pfss_get_potl_coeffs
def  pfss_get_potl_coeffs (mag,rtop,quiet):
    #if len(mag) == 0:
        #print,'  pfss_get_potl_coeffs,mag,rtop=rtop,/quiet'
        #return
    
    cth = np.cos(theta)
    sth=np.sqrt(1-cth*cth)
    nlat=len(cth)
    nlon=2*nlat
    nax = np.shape(mag)
    if nlon != nax[0]:
        print('ERROR in pfss_get_potl_coeffs:  nlon is off',nlon)
        return
    if nlat != nax[1]:
        print('ERROR in pfss_get_potl_coeffs:  nlat is off')
        return
    #get spherical harmonic transform of magnetogram
    lmax=nlat
    magt=spherical_transform(mag,cth,lmax, None)

    #get l and m index arrays of transform
    lix=np.arange(0,lmax+1)
    mix=lix
    larr=lix * (np.repeat(1,lmax+1))[:, np.newaxis]
    marr=np.repeat(1,lmax+1)* mix[:, np.newaxis]
    wh=np.nonzero(marr > larr)
    larr[wh]=0
    marr[wh]=0

    #determine coefficients
    if rtop is not None: # source surface at rtop
        rtop=np.double(rtop)
        phibt=-magt/(1+larr*(1+rtop**(-2*larr-1)))
        phiat=-phibt/(rtop**(2*larr+1))
        wh=np.nonzero(np.isfinite(phiat) = 0)
        nwh = np.count_nonzero(wh)
        if nwh > 0:
            phiat[wh]=complex(0,0)
    else: #potential field extends to infinity, all A's are 0
        phiat=magt
        phiat[:]=0
        phibt=-magt/(1+larr)
    
    if quiet is None:     
        print ('pfss_get_potl_coeffs:  forward transform completed')
    path='C:/Users/amaro/Desktop/Roksolanai/'
    with open(path+"/19990513/MDI_0000__PFSS_0004/pfss_data_block.sav", 'wb') as f:  
        np.save(f, phiat)
        np.save(f, phibt)
    
#####################################################################
def pfss_print_time (string,it,nit,tst,slen,elapsed):
    import time
    #sets tst if not set
    if ((tst is None) or (it == 1)):
        tst=np.long(time.process_time())#time.perf_counter or time.process_time
    #calculate time remaining in minutes (or seconds if under 1 minute)
    telapsed=np.double(np.long(time.process_time())-tst)  #time elapsed (seconds)
    if it > 1: #avoids division by zero errors
        if elapsed is not None:
            time=telapsed
        else:
            itrate=telapsed/(it-1)  # current iteration rate (iterations per second)
            time=(nit-it+1)*itrate  # time remaining (seconds)
        if time < 59.5:
            time=round(time)
            label=' second'
        else:
            time=round(time/60.)
            label=' minute'
        timestr=str(time)+label
        if time != 1:
            timestr=timestr+'s'
    else:
        timestr = '---'
        if elapsed is not None:
            timestr='elapsed = '+timestr 
        else:
            timestr='left = '+timestr
            
    #  erase previous message
    if len(slen) > 0:
        print(str(np.uint8(13))+'($,A)')

    #print time left
    outstr=string+'on iteration '+str(it)+' of '+str(nit)+', time '+timestr+'     '
    print(outstr+'($,A)')

    #  set slen variable
    slen=len(str(outstr))

    # print if on final iteration
    if it == nit:
        print()


def inv_spherical_transform (B,cp,lmax,mrange,period,phirange,cprange,thindex,phindex):
    #preliminaries
    if period is not None:
        period=round(period[0])
    else:
        period=1
    if lmax is not None:
        lmax=round(np.long(lmax[0]))
    else:
        lmax=len(B[:][0])-1
    ############################################################
    def mrange0 ():
        mrange=[0,lmax]
    def mrange1 ():
        mrange_0 =(round(mrange[0])<lmax)
        if mrange_0 is True:
            mrange_0 = round(mrange[0])
        else:
            mrange_0 = lmax
        mrange=[0,mrange_0]
    def default ():
        mrange_1 =(round(mrange[1])<lmax)
        if mrange_1 is True:
            mrange_1 = round(mrange[1])
        else:
            mrange_1 = lmax
        mrange=[round(mrange[0]),mrange_1]

    case = {
            0 : mrange0,
            1 : mrange1     
           }

    def mrange_function(mrange):
        return case.get(len(mrange), default)()
    
    ###########################################################
    
    #determine output (co-)latitude grid
    if len(thindex) > 0:
        if max(thindex) > np.pi:
            print ('  ERROR in inv_spherical_transform: thindex out of range')
            return -1
        elif min(thindex)< np.float64(0e0):
            print ('  ERROR in inv_spherical_transform: thindex out of range')
            return -1
        ntheta=len(thindex)
        costheta=np.cos(np.double(thindex))
        sintheta=np.sqrt(1-costheta*costheta)
    else:
        def cprange0():
            cp1i=-1.0
            cp2i=1.0
        def cprange1():
            cp2i=abs(cprange[0])<1
            if cp2i is True:
                cp2i = abs(cprange[0])
            else:
                cp2i = 1

            cp1i=-cp2i
        def default ():
            cp1i=min(cprange)>(-1)

            if cp1i is True:
                cp1i = min(cprange)
            else:
                cp1i = (-1)
                
            cp2i=max(cprange)<1
            
            if cp2i is True:
                cp2i = max(cprange)
            else:
                cp2i = 1
            
        case2 = {
        0 : cprange0,
        1 : cprange1     
        }

        def cprange_function(cprange):
            return case2.get(len(cprange), default)()

        wh=np.nonzero((cp >= cp1i) and (cp <= cp2i))
        ntheta = np.count_nonzero(wh)
        if ntheta is False:
            print ('  ERROR in inv_spherical_transform: invalid cprange')
            return -1
        costheta=np.double(cp[wh])
        sintheta=np.sqrt(1-costheta*costheta)
        thindex=np.acos(costheta)

    #determine output longitude grid
    if len(phindex) > 0:
        nphi=len(phindex)
        phiix=np.double(phindex)
    else:
        nphi=2*len(cp)
        phiix=2*np.pi*np.arange(nphi/period)/nphi
        
        def phirange0():
            ph1i=0.0
            ph2i=2*np.pi/period
            
        def phirange1():
            ph1i=0.0
            ph2i=(2*np.pi/period)<phirange[0]
            if ph2i is True:
                ph2i = 2*np.pi/period
            else:
                ph2i = phirange[0]
                
        def default ():
            ph1i=0.0>min(phirange)
            ph2i=(2*np.pi/period)<max(phirange)
            
            if ph2i is True:
                ph2i = 2*np.pi/period
            else:
                ph2i = max(phirange)
            
        case3 = {
        0 : phirange0,
        1 : phirange1     
        }

        def cprange_function(phirange):
            return case3.get(len(phirange), default)()
    
        wh=np.nonzero((phiix >=ph1) and (phiix <= ph2i))
        nphi = np.count_nonzero(wh)
        if nphi is False:
            print ('  ERROR in inv_spherical_transform: invalid phirange')
            return -1
        phiix=phiix[wh]
            
    #calculate array of amplitudes and phases of B
    Bamp=abs(B)
    z = complex(B)
    phase=np.atan(z.imag,np.double(B))
    
    #set up output array A
    A=np.zeros(nphi,ntheta)
    #take care of modes where m=0
    CP_0_0=1/np.sqrt(4*np.pi)
    if mrange[0] == 0:
        #start with m=l=0 mode
        A=A+np.repeat(Bamp[0][0]*np.cos(phase[0][0])*CP_0_0,[nphi,ntheta])

        #  now do l=1 m=0 mode
        CP_1_0=np.sqrt(3.e0)*costheta*CP_0_0
        Y=np.repeat(np.cos(phase[1][0]),nphi) * CP_1_0[:, np.newaxis]
        A=A+(Bamp[1][0]*Y)

        #  do other l modes for which m=0
        if lmax > 1:
            CP_lm1_0=CP_0_0
            CP_l_0=CP_1_0
            for l in range (2,lmax+1):
                ld=np.double(l)
                CP_lm2_0=CP_lm1_0
                CP_lm1_0=CP_l_0
                c1=np.sqrt(4*ld^2-1)/ld
                c2=np.sqrt((2*ld+1)/(2*ld-3))*((ld-1)/ld)
                CP_l_0=c1*costheta*CP_lm1_0-c2*CP_lm2_0
                Y=np.repeat(np.cos(phase[l][0]),nphi) * CP_l_0[:, np.newaxis]
                A=A+(Bamp[l][0]*Y)
       
    #loop through m's for m>0, and then loop through l's for each m
    CP_m_m=CP_0_0
    for m in range (1,mrange[1]+1):
        md=np.double(m)

        #do l=m mode first
        CP_mm1_mm1=CP_m_m
        CP_m_m=-np.sqrt(1+1/(2*md))*sintheta*CP_mm1_mm1
        if (mrange[0] <= m) and ((m % period) == 0):
            angpart=np.cos(md*phiix + phase[m][int(m/period)])
            A=A+Bamp[m][int(m/period)]*(angpart*CP_m_m[:, np.newaxis])

            #  now do l=m+1 mode
            if lmax >= m+1 :
                CP_mp1_m=np.sqrt(2*md+3)*costheta*CP_m_m
                angpart=np.cos(md*phiix+phase[m+1][int(m/period)])
                A=A+Bamp[m+1][int(m/period)]*(angpart*CP_mp1_m[:, np.newaxis])
            #now do other l's
            if lmax >= m+2:
                CP_lm1_m=CP_m_m
                CP_l_m=CP_mp1_m
                for l in range (m+2,lmax+1):
                    ld=np.double(l)
                    CP_lm2_m=CP_lm1_m
                    CP_lm1_m=CP_l_m
                    c1=np.sqrt((4*ld^2-1)/(ld^2-md^2))
                    c2=np.sqrt(((2*ld+1)*((ld-1)^2-md^2))/((2*ld-3)*(ld^2-md^2)))
                    CP_l_m=c1*costheta*CP_lm1_m-c2*CP_lm2_m
                    angpart=np.cos(md*phiix+phase[l][int(m/period)])
                    A=A+Bamp[l][int(m/period)]*(angpart*CP_l_m[:, np.newaxis])
   
    return A            
            
    
#Check pfss_get_potl_coeffs
def  pfss_potl_field__AV (rtop,rgrid,rindex,thindex,phindex,lmax,trunc,potl,quiet):

    #preliminaries
    nlat0 = nlat
    if lmax is not None:
        lmax = lmax
    else:
        lmax = nlat0
    cth = np.cos(theta)
    #get l and m index arrays of transform
    phisiz=np.shape(phiat)
    lix=np.arange(0,np.ndim(phiat))
    mix=np.arange(0,phisiz[0])  
    larr=lix * (np.repeat(1,phisiz[0]))[:, np.newaxis]
    marr=np.repeat(1,np.ndim(phiat))* mix[:, np.newaxis]
    wh=np.nonzero(marr > larr)
    larr[wh]=0
    marr[wh]=0
    
    #get radial grid
    dr0=[np.pi/nlat0]  #r grid spacing at r=1, make it half avg lat grid spacing
    rra=[np.float64(1e0),np.double(rtop[0])]  #????  #range of r
    if len(rgrid) == 0:
        rgrid = 1
    #######################################################    
    def rgrid2():
        rix=[rra[0]]
        lastr=rra[0]
        while True:
            nextr=lastr+dr0*(lastr/rra[0])**2
            rix=[rix][nextr]
            lastr=nextr
            if nextr >= rra[1]: break
        rix2=rix/((max(rix)-rra[0])/(rra[1]-rra[0]))
        rix=rix2+(rra[0]-rix2[0])
        nr=len(rix)
        
    def rgrid3():
        if len(rindex) == 0:
            print,'  ERROR in pfss_potl_field: rindex must be set if rgrid=3'
            return
        rix=rindex
        nr=len(rindex)
        
    def default ():
        nr=round((rra[1]-rra[0])/dr0)
        rix=np.linspace(rra[0],rra[1],nr)
        
    switcher = {
        # radial gridpoint separation is proportional to r^2
        2 : rgrid2,
        # custom radial grid
        3 : rgrid3      
        }

    def rgrid_function(rgrid):
        return switcher.get(rgrid, default)()
    ###############################################################
                         
    if quiet is None:
        print ('pfss_potl_field: nr =',nr)

    #set up theta grids
    if len(thindex) > 0:
        if max(thindex) > np.pi:
            print ( 'ERROR in pfss_potl_field: thindex out of range')
            return
        elif min(thindex) < np.float64(0e0):
            print ('  ERROR in pfss_potl_field: thindex out of range')
            return
        ntheta=len(thindex)
        theta=thindex
        nlat=ntheta
        lat=90-theta*180/np.pi
    else:
        ntheta=nlat
        thindex=theta

    #set up phi grid
    if len(phindex) > 0:
        nphi=len(phindex)
        phi=phindex
        nlon=nphi
        lon=phi*180/np.pi
    else:
        nphi=nlon
        phindex=phi

    #set up planar sin(theta) array
    stharr=np.repeat(1,nphi)*(np.sqrt(1-(np.cos(thindex))**2))[:, np.newaxis]

    #compute lmax for each radius
    lmaxarr=np.zeros(nr,dtype=object)
    
    if trunc is not None: #include fewer l modes as you get higher up
        lmaxarr[0]=lmax
        for i in range (1,nr):
            wh=np.nonzero(rix[i]**np.arange(0,lmax+1) > 1e6)
            nwh = np.count_nonzero(wh)
            """ wh = (rix[i]**np.arange(0,lmax+1) > 1e6).nonzero()
		wh = wh[0]
		nwh = len(wh)"""
            if nwh == 0:
                lmaxarr[i]=lmax
            else:
                lmaxarr_0 = (wh[0]<lmax)
                if lmaxarr_0 is True:
                    lmaxarr[i] = wh[0]
                else:
                    lmaxarr[i] = lmax
    else:
        lmaxarr_1 = nlat0<lmax
        if lmaxarr_1 is True:
            lmaxarr[:] = nlat0
        else:
            lmaxarr[:] = lmax  #otherwise do nlat transforms for all radii

    #compute Br in (r,l,m)-space
    bt=np.zeros((phisiz,nr),dtype = complex)
    for i in range (0,nr):
        bt[:][:][i]= phiat*larr*rix[i]**(larr-1) - phibt*(larr+1)*rix[i]**(-larr-2)

    # ...and then transform to (r,theta,phi)-space
    br=np.zeros((nphi,ntheta,nr),dtype = float)
    for i in range (0,nr):
        if quiet is None:
            pfss_print_time('pfss_potl_field: computing Br: ',i+1,nr,tst,slen, None)
        br[:][:][i]=inv_spherical_transform(bt[:][:][i],cth,lmaxarr[i],None, None, None, None, thindex,phindex)/stharr
    #compute sin(theta) * Bth in (r,l,m)-space...
    factor=np.sqrt(np.double(larr**2-marr**2)/np.double(4*larr**2-1))

    for i in range (0,nr):
        br[:][:][i]=(larr-1)*factor*(np.roll(phiat,1,0)*rix[i]**(larr-2) + snp.roll(phibt,1,0)*rix[i]**(-larr-1))
        - (larr+2)*np.roll(factor,-1,0)* (np.roll(phiat,-1,0)*rix[i]**larr + np.roll(phibt,-1,0)*rix[i]**(-larr-3))
        bt[0][0][i]=-2*factor[1][0]*(phiat[1][0] + phibt(1,0)*rix[i]**(-3))
        bt[lmax][:][i]=(lmax-1)*factor[lmax][:]*(phiat[lmax-1][:]*rix[i]**(lmax-2) + phibt[lmax-1][:]*rix[i]**(-lmax-1))
    
    #...and then compute Bth in (r,theta,phi)-space
    bth=np.zeros((nphi,ntheta,nr),dtype = float)
    for i in range (0,nr):
        if quiet is None:
             pfss_print_time('  pfss_potl_field: computing Bth:  ',i+1,nr,tst,slen,None)
             
        bth[:][:][i]=inv_spherical_transform(bt[:][:][i],cth,lmaxarr[i],None, None, None, None, thindex,phindex)/stharr

    #compute sin(theta) * Bph in (r,l,m)-space...
    for i in range (0,nr):
        bt[:][:][i]=np.complex(0,1)*marr*(phiat*rix[i]**(larr-1) + phibt*rix[i]**(-larr-2))

    # ...and then compute Bph in (r,theta,phi)-space
    bph=np.zeros((nphi,ntheta,nr),dtype = float)

    for i in range (0,nr):
        if quiet is None:
             pfss_print_time('  pfss_potl_field: computing Bph:  ',i+1,nr,tst,slen,None)
               
        bph[:][:][i]=inv_spherical_transform(bt[:][:][i],cth,lmaxarr[i],None, None, None, None, thindex,phindex)/stharr

     #now transform the field potential to (r,theta,phi)-space
    if len(potl) > 0:
        potl=np.zeros((nlon,nlat,nr),dtype = float)
        for i in range (0,nr):
            if quiet is None:
                pfss_print_time('  pfss_potl_field: computing the field potential:  ',i+1,nr,tst,slen,None)

            potl[:][:][i]=inv_spherical_transform(phibt*rix[i]**(-larr-1)+ phiat*rix[i]**larr,cth,lmax,None, None, None, None, None, None)
            
  


        
def mean_dtheta (A, costheta):
    #preliminaries
    naxin=np.shape(A)
    ndim=len(naxin)
    x=np.double(costheta)
    nx=len(x)
    #set output axes
    if len(naxin) == 1:
        naxout=np.int32(1)
    else:
        naxout=naxin[1:]

    #error checking
    if nx != naxin[0]:
        print('  mean_dtheta.pro:  size of costheta and first dim of A do not agree') 
        return -1

    #reform input array
    dim2=np.int32(1)
    if ndim > 1:
        for i in range (1,ndim):
            dim2=dim2*naxin[i]
    nax=[nx,dim2]
    #AA=transpose(reform(A,nax))
    A.shape=(nax)
    AA=np.transpose(A)
    
    #get integration weights
    weights=weights_legendre(costheta)/(4*np.pi)

    #integrate
    #result=reform(weights##AA,naxout)
    (weights[:, np.newaxis]*AA).shape = (naxout)
    result=(weights[:, np.newaxis]*AA)

    return result
   
###transpose reform    
### #??? a1* a2[:, numpy.newaxis]            
#s = numpy.transpose(d)
#d.shape = (2,3) 2 - rindasn3 kolonnas
        
        
def pfss_potl_field (rtop,rgrid,rindex,thindex,phindex,lmax,trunc,potl,quiet):

    #preliminaries
    nlat0 = nlat
    if lmax is not None:
        lmax_2 = lmax<nlat0
        if lmax_2 is True:
            lmax = lmax
        else:
            lmax = nlat0
    else:
        lmax = nlat0

    cth = np.cos(theta)
    #get l and m index arrays of transform
    
    phisiz=size(phiat,/dim)
    lix=lindgen(phisiz(0))
    mix=lindgen(phisiz(1))
    larr=lix#replicate(1,phisiz(1))
    marr=replicate(1,phisiz(0))#mix
    wh=where(marr gt larr)
    larr(wh)=0  &  marr(wh)=0
    
    
    
    
