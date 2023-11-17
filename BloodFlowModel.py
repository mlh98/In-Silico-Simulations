import numpy as np
import matplotlib.pyplot as plt


def p(z,t,v):
    """Laminar isotropic pipe flow. Distribution is transformed from 
      velocity distribution p(v) to spatial distribution p(z)."""
    return np.where(np.absolute(z/t) < 2*v, 1/(4*v*t)*np.log(2*v/np.absolute(z/t)), 0)


def define_slice_and_gap(d, g, n_slices, delta_z):
    """
    Define slice and gap positions. 
    
    Parameters:
        d: slice thickness
        g: distance between slices
        n_slices: # of slices
        delta_z: Step size for sampled z-positions.
    
    Returns: 
        slice_number: all enumerated slices
        slices_even: enumerated even slices
        slices_odd: enumerated odd slices
        z_slice0: Array with values in central slice
        z_gap0: Array with gap below central slice
        slice_positions_even: Array of positions of even slices
        slice_positions_odd: Array of positions of odd slices
        gap_positions: Array of positions of all slice gaps
    """
    
    # Define central slice from -d/2 to d/2 and central gap below the central
    # slice
    z_slice0 = np.arange(-d/2, d/2, delta_z)
    z_gap0 = np.arange(-d/2-g, -d/2, delta_z)
    
    # Differentiate between even and odd slices
    if n_slices % 2 == 0:
        slice_number = np.arange(-n_slices/2+1, n_slices/2+1, 1)
    else:
        slice_number = np.arange(-n_slices/2+1, n_slices/2+1, 1) - 0.5
    slice_number = slice_number.astype(np.int32)
    slices_even = slice_number[slice_number % 2 == 0]
    slices_odd = slice_number[slice_number % 2 != 0]
    
    # Initialize even and odd slice positions and gap positions
    slice_positions_even = np.zeros((np.size(slices_even),np.size(z_slice0)))
    slice_positions_odd = np.zeros((np.size(slices_odd),np.size(z_slice0)))
    gap_positions = np.zeros((np.size(slice_number),np.size(z_gap0)))
    
    # Loop over all (even and odd) slices and gaps
    for i,n in enumerate(slices_even):
        slice_positions_even[i,:] = z_slice0+n*(d+g)
    for i,n in enumerate(slices_odd):
        slice_positions_odd[i,:] = z_slice0+n*(d+g)
    for i,n in enumerate(slice_number):
        gap_positions[i,:] = z_gap0+n*(d+g)
        
    return slice_number, slices_even, slices_odd, z_slice0, z_gap0, slice_positions_even, slice_positions_odd, gap_positions


def r_pipe(z, d, g, n_slices, delta_z, v, TR, T1_blood, interleaved=False):
    """
    Compute relative blood signal
    
    Parameters:
        z: Spatial position of the central slice (slice of interest)
        d: Slice thickness
        g: Distance between slices
        n_slices: # of slices
        delta_z: Step size for sampled z-positions.
        v: Blood flow velocity
        TR: Repetition time
        T1_blood: T1 relaxation time of blood
        interleaved=True: Interleaved slice acquisition
        interleaved=False: Ascending slice acquisition
    
    Returns:
        W_sts_even: Relative amount of blood flowing from even slices to the 
                    central slice
        W_sts_odd: Relative amount of blood flowing from odd slices to the 
                   central slice
        W_gts: Relative amount of blood flowing from gaps to the central
               slice
        Relative blood signal (weighted with T1 relaxation) depending on 
        interleaved or ascending slice acquisition mode
    """
    
    # Compute all slice and gap positions
    slice_number, slices_even, slices_odd, z_slice0, z_gap0, slice_positions_even, slice_positions_odd, gap_positions = define_slice_and_gap(d, g, n_slices, delta_z)
    
    # Initialize relative amount of blood traveling from slice to slice (W_sts)
    # and from gap to slice (W_gts)
    W_sts_even = 0
    W_sts_odd = 0
    W_gts = 0

    # Loop over even slices
    for slices,i in enumerate(slices_even):
        # Loop over all initial starting points of blood magnetization in even 
        # slice positions
        for z0 in slice_positions_even[slices,:]:
            # Loop over all final positions of flowing blood magnetization in 
            # the central slice
            for z_final in z:
                # Compute relative amount of blood flowing from even slices to 
                # the central slice
                W_sts_even = W_sts_even + p(z_final+z0,TR,v)
    
    # Loop over odd slices
    for slices,i in enumerate(slices_odd):
        # Loop over all initial starting points of blood magnetization in odd 
        # slice positions
        for z0 in slice_positions_odd[slices,:]:
            # Loop over all final positions of flowing blood magnetization in 
            # the central slice
            for z_final in z:
                # Distinguish between interleaved and ascending slice 
                # acquisition mode
                if interleaved==True:
                    # Compute relative amount of blood flowing from odd slices 
                    # to the central slice; Magnetization has already seen an
                    # excitation pulse in the odd slice TR/2 ago
                    W_sts_odd = W_sts_odd + p(z_final+z0,TR/2,v)
                else:
                    # Compute relative amount of blood flowing from odd slices 
                    # to the central slice; Ascending mode = blood is saturated
                    # TR after the last pulse
                    W_sts_odd = W_sts_odd + p(z_final+z0,TR,v)

    # Loop over gaps
    for gap in slice_number:
        # Loop over all initial starting points of blood magnetization in gaps
        for z0 in gap_positions[gap,:]:
            # Loop over all final positions of flowing blood magnetization in 
            # the central slice
            for z_final in z:
                # Compute relative amount of blood flowing from gap positions 
                # to the central slice
                W_gts = W_gts + p(z_final+z0,TR,v)
    
    # Compute the total relative amount of blood from all starting positions
    # for normalization
    W_tot = W_gts + W_sts_even + W_sts_odd
    
    # Normalize
    W_sts_even = W_sts_even/W_tot
    W_sts_odd = W_sts_odd/W_tot
    W_gts = W_gts/W_tot
    
    if interleaved==True:
        return W_sts_even, W_sts_odd, W_gts, W_gts + W_sts_even*(1-np.exp(-TR/T1_blood)) + W_sts_odd*(1-np.exp(-TR/2/T1_blood))
    else:
        return W_sts_even, W_sts_odd, W_gts, W_gts + (W_sts_even + W_sts_odd)*(1-np.exp(-TR/T1_blood))


def perfusion_fraction(z, d, g, n_slices, delta_z, TR, T1_blood, T1_liver, v, 
                       f_meas, TR_meas, interleaved=False):
    """
    Compute perfusion fraction.
    
    Parameters:
        z: Spatial position of the central slice (slice of interest)
        d: Slice thickness
        g: Distance between slices
        n_slices: # of slices
        delta_z: Step size for sampled z-positions.
        TR: Repetition time
        T1_blood: T1 relaxation time of blood
        T1_liver: T1 relaxation time of liver tissue
        v: array of velocities
        f_meas: Measured f at given TR as reference
        TR_meas: TR of f_meas
        interleaved=True: Interleaved slice acquisition
        interleaved=False: Ascending slice acquisition
        
    Returns:
        Array of computed perfusion fraction for each velocity v
    """
    
    # Initialize relative amount of blood and relative blood signals
    W_sts_even, W_sts_odd, W_gts = np.zeros(n_v), np.zeros(n_v), np.zeros(n_v)
    r = np.zeros(n_v)
    
    # Initialize relative amount of blood and relative blood signals for 
    # TR_meas
    W_sts_even_TR_meas, W_sts_odd_TR_meas, W_gts_TR_meas = np.zeros(n_v), np.zeros(n_v), np.zeros(n_v)
    r_TR_meas = np.zeros(n_v)
    
    # Loop over velocities
    for i,v0 in enumerate(v):
        print("\n{}/{}".format(i+1, n_v))
        print("v0 = ", v0, "mm/s")
        W_sts_even[i], W_sts_odd[i], W_gts[i], r[i] = r_pipe(z,d,g,n_slices,delta_z,v0,TR,T1_blood,interleaved)
        W_sts_even_TR_meas[i], W_sts_odd_TR_meas[i], W_gts_TR_meas[i], r_TR_meas[i] = r_pipe(z,d,g,n_slices,delta_z,v0,TR_meas,T1_blood,interleaved)
        
    # Compute f_infinity (perfusion fraction for TR>>T1)
    f_inf_pipe = ((np.exp(-TR_meas/T1_liver)-1)*f_meas)/(-r_TR_meas + f_meas*(np.exp(-TR_meas/T1_liver)-1+r_TR_meas))

    return r/(r + (1/f_inf_pipe-1)*(1-np.exp(-TR/T1_liver)))


# Example: Comparison between short and long TR for 5 mm slice thickness
# Parameters
n_v = 30
v = np.linspace(0.1, 15, n_v)
TR = 1.3
TR2 = 4.5
TR_meas = 4.5
T1_liver = 0.810
T1_blood = 1.3
f_meas = 0.28

# slices and gaps
n_slices = 9
d = 5
g = 5

delta_z = 0.1
# p(z) is not defined at z=0 -> +0.0001
z = np.arange(-d/2, d/2, delta_z)+0.0001

# Perfusion fractions for short and long TR
f = perfusion_fraction(z, d, g, n_slices, delta_z, TR, T1_blood, T1_liver, v, 
                       f_meas, TR_meas, interleaved=False)
f_TR2 = perfusion_fraction(z, d, g, n_slices, delta_z, TR2, T1_blood, T1_liver, 
                           v, f_meas, TR_meas, interleaved=False)


# Plot commands
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel("$v_0$ [mm/s]", fontsize=20)
ax.set_ylabel("f [%]", fontsize=20)
ax.tick_params(axis='both', labelsize=20, width=3)

ax.plot(v, f*100, color='red', linestyle='dotted',
        label='{:.0f} mm {:.0f} ms'.format(d, TR*1000), 
        linewidth=2.5)
ax.plot(v, f_TR2*100, color='red', linestyle='dashdot',
        label='{:.0f} mm {:.0f} ms'.format(d, TR2*1000), 
        linewidth=2.5) 

ax.legend()

plt.show() 
