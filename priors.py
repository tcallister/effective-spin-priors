import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import spence as PL

def Di(z):

    """
    Wrapper for the scipy implmentation of Spence's function.
    Note that we adhere to the Mathematica convention as detailed in:
    https://reference.wolfram.com/language/ref/PolyLog.html

    Inputs
    z: A (possibly complex) scalar or array

    Returns
    Array equivalent to PolyLog[2,z], as defined by Mathematica
    """

    return PL(1.-z+0j)

def chi_effective_prior_from_aligned_spins(q,aMax,xs):

    """
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, aligned component spin priors.

    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior

    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(xs,-1)

    # Set up various piecewise cases
    pdfs = np.zeros(xs.size)
    caseA = (xs>aMax*(1.-q)/(1.+q))*(xs<=aMax)
    caseB = (xs<-aMax*(1.-q)/(1.+q))*(xs>=-aMax)
    caseC = (xs>=-aMax*(1.-q)/(1.+q))*(xs<=aMax*(1.-q)/(1.+q))

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]

    # Handle vectorized q
    if np.asarray([q]).size == 1:
        q_A = q
        q_B = q
        q_C = q
    else:
        q_A = q[caseA]
        q_B = q[caseB]
        q_C = q[caseC]

    if np.sum(caseA) > 0:
        pdfs[caseA] = (1. + q_A)**2 * (aMax - x_A) / (4. * q_A*aMax**2)
    if np.sum(caseB) > 0:
        pdfs[caseB] = (1. + q_B)**2 * (aMax - x_B) / (4. * q_B*aMax**2)
    if np.sum(caseC) > 0:
        pdfs[caseC] = (1.+q_C)/(2.*aMax)

    return pdfs

def chi_effective_prior_from_isotropic_spins(q,aMax,xs):

    """
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, isotropic component spin priors.

    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior

    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(np.abs(xs),-1)

    # Set up various piecewise cases
    pdfs = np.ones(xs.size,dtype=complex)*(-1.)
    caseZ = (xs==0)
    caseA = (xs>0)*(xs<aMax*(1.-q)/(1.+q))*(xs<q*aMax/(1.+q))
    caseB = (xs<aMax*(1.-q)/(1.+q))*(xs>q*aMax/(1.+q))
    caseC = (xs>aMax*(1.-q)/(1.+q))*(xs<q*aMax/(1.+q))
    caseD = (xs>aMax*(1.-q)/(1.+q))*(xs<aMax/(1.+q))*(xs>=q*aMax/(1.+q))
    caseE = (xs>aMax*(1.-q)/(1.+q))*(xs>aMax/(1.+q))*(xs<aMax)
    caseF = (xs>=aMax)

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]
    x_D = xs[caseD]
    x_E = xs[caseE]

    # Handle vectorized q
    if np.asarray([q]).size == 1:
        q_A = q
        q_B = q
        q_C = q
        q_D = q
        q_E = q
        q_Z = q
    else:
        q_A = q[caseA]
        q_B = q[caseB]
        q_C = q[caseC]
        q_D = q[caseD]
        q_E = q[caseE]
        q_Z = q[caseZ]

    if len(caseZ) > 0:
        pdfs[caseZ] = (1.+q_Z)/(2.*aMax)*(2.-np.log(q_Z))

    if len(caseA) > 0:
        pdfs[caseA] = (1.+q_A)/(4.*q_A*aMax**2)*(
                    q_A*aMax*(4.+2.*np.log(aMax) - np.log(q_A**2*aMax**2 - (1.+q_A)**2*x_A**2))
                    - 2.*(1.+q_A)*x_A*np.arctanh((1.+q_A)*x_A/(q_A*aMax))
                    + (1.+q_A)*x_A*(Di(-q_A*aMax/((1.+q_A)*x_A)) - Di(q_A*aMax/((1.+q_A)*x_A)))
                    )

    if len(caseB) > 0:
        pdfs[caseB] = (1.+q_B)/(4.*q_B*aMax**2)*(
                    4.*q_B*aMax
                    + 2.*q_B*aMax*np.log(aMax)
                    - 2.*(1.+q_B)*x_B*np.arctanh(q_B*aMax/((1.+q_B)*x_B))
                    - q_B*aMax*np.log((1.+q_B)**2*x_B**2 - q_B**2*aMax**2)
                    + (1.+q_B)*x_B*(Di(-q_B*aMax/((1.+q_B)*x_B)) - Di(q_B*aMax/((1.+q_B)*x_B)))
                    )

    if len(caseC) > 0:
        pdfs[caseC] = (1.+q_C)/(4.*q_C*aMax**2)*(
                    2.*(1.+q_C)*(aMax-x_C)
                    - (1.+q_C)*x_C*np.log(aMax)**2.
                    + (aMax + (1.+q_C)*x_C*np.log((1.+q_C)*x_C))*np.log(q_C*aMax/(aMax-(1.+q_C)*x_C))
                    - (1.+q_C)*x_C*np.log(aMax)*(2. + np.log(q_C) - np.log(aMax-(1.+q_C)*x_C))
                    + q_C*aMax*np.log(aMax/(q_C*aMax-(1.+q_C)*x_C))
                    + (1.+q_C)*x_C*np.log((aMax-(1.+q_C)*x_C)*(q_C*aMax-(1.+q_C)*x_C)/q_C)
                    + (1.+q_C)*x_C*(Di(1.-aMax/((1.+q_C)*x_C)) - Di(q_C*aMax/((1.+q_C)*x_C)))
                    )

    if len(caseD) > 0:
        pdfs[caseD] = (1.+q_D)/(4.*q_D*aMax**2)*(
                    -x_D*np.log(aMax)**2
                    + 2.*(1.+q_D)*(aMax-x_D)
                    + q_D*aMax*np.log(aMax/((1.+q_D)*x_D-q_D*aMax))
                    + aMax*np.log(q_D*aMax/(aMax-(1.+q_D)*x_D))
                    - x_D*np.log(aMax)*(2.*(1.+q_D) - np.log((1.+q_D)*x_D) - q_D*np.log((1.+q_D)*x_D/aMax))
                    + (1.+q_D)*x_D*np.log((-q_D*aMax+(1.+q_D)*x_D)*(aMax-(1.+q_D)*x_D)/q_D)
                    + (1.+q_D)*x_D*np.log(aMax/((1.+q_D)*x_D))*np.log((aMax-(1.+q_D)*x_D)/q_D)
                    + (1.+q_D)*x_D*(Di(1.-aMax/((1.+q_D)*x_D)) - Di(q_D*aMax/((1.+q_D)*x_D)))
                    )

    if len(caseE) > 0:
        pdfs[caseE] = (1.+q_E)/(4.*q_E*aMax**2)*(
                    2.*(1.+q_E)*(aMax-x_E)
                    - (1.+q_E)*x_E*np.log(aMax)**2
                    + np.log(aMax)*(
                        aMax
                        -2.*(1.+q_E)*x_E
                        -(1.+q_E)*x_E*np.log(q_E/((1.+q_E)*x_E-aMax))
                        )
                    - aMax*np.log(((1.+q_E)*x_E-aMax)/q_E)
                    + (1.+q_E)*x_E*np.log(((1.+q_E)*x_E-aMax)*((1.+q_E)*x_E-q_E*aMax)/q_E)
                    + (1.+q_E)*x_E*np.log((1.+q_E)*x_E)*np.log(q_E*aMax/((1.+q_E)*x_E-aMax))
                    - q_E*aMax*np.log(((1.+q_E)*x_E-q_E*aMax)/aMax)
                    + (1.+q_E)*x_E*(Di(1.-aMax/((1.+q_E)*x_E)) - Di(q_E*aMax/((1.+q_E)*x_E)))
                    )

    if len(caseF) > 0:
        pdfs[caseF] = 0.

    # Deal with spins on the boundary between cases
    if np.any(pdfs==-1): # Not tested by Vera
        boundary = (pdfs==-1)
        pdfs[boundary] = 0.5*(chi_effective_prior_from_isotropic_spins(q[boundary],aMax,xs[boundary]+1e-6)\
                        + chi_effective_prior_from_isotropic_spins(q[boundary],aMax,xs[boundary]-1e-6))

    return np.real(pdfs)

def chi_p_prior_from_isotropic_spins(q,aMax,xs):

    """
    Function defining the conditional priors p(chi_p|q) corresponding to
    uniform, isotropic component spin priors.

    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_p value or values at which we wish to compute prior

    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(xs,-1)

    # Set up various piecewise cases
    pdfs = np.zeros(xs.size)
    caseA = xs<q*aMax*(3.+4.*q)/(4.+3.*q)
    caseB = (xs>=q*aMax*(3.+4.*q)/(4.+3.*q))*(xs<aMax)

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]

    # Handle vectorized q
    if np.asarray([q]).size == 1:
        q_A = q
    else:
        q_A = q[caseA]


    if np.sum(caseA) > 0:
        pdfs[caseA] = (1./(aMax**2*q_A))*((4.+3.*q_A)/(3.+4.*q_A))*(
                    np.arccos((4.+3.*q_A)*x_A/((3.+4.*q_A)*q_A*aMax))*(
                        aMax
                        - np.sqrt(aMax**2-x_A**2)
                        + x_A*np.arccos(x_A/aMax)
                        )
                    + np.arccos(x_A/aMax)*(
                        aMax*q_A*(3.+4.*q_A)/(4.+3.*q_A)
                        - np.sqrt(aMax**2*q_A**2*((3.+4.*q_A)/(4.+3.*q_A))**2 - x_A**2)
                        + x_A*np.arccos((4.+3.*q_A)*x_A/((3.+4.*q_A)*aMax*q_A))
                        )
                    )
                    
    if np.sum(caseB) > 0:
        pdfs[caseB] = (1./aMax)*np.arccos(x_B/aMax)

    return pdfs

def joint_prior_from_isotropic_spins(q,aMax,xeffs,xps,ndraws=10000,bw_method='scott'):

    """
    Function to calculate the conditional priors p(xp|xeff,q) on a set of {xp,xeff,q} posterior samples.

    INPUTS
    q: Mass ratio
    aMax: Maximimum spin magnitude considered
    xeffs: Effective inspiral spin samples
    xps: Effective precessing spin values
    ndraws: Number of draws from the component spin priors used in numerically building interpolant

    RETURNS
    p_chi_p: Array of priors on xp, conditioned on given effective inspiral spins and mass ratios
    """

    # Convert to arrays for safety
    xeffs = np.reshape(xeffs,-1)
    xps = np.reshape(xps,-1)
    assert len(np.asarray([q])) == 0
    
    # Compute marginal prior on xeff, conditional prior on xp, and multiply to get joint prior!
    p_chi_eff = chi_effective_prior_from_isotropic_spins(q,aMax,xeffs)
    p_chi_p_given_chi_eff = np.array([chi_p_prior_given_chi_eff_q(q,aMax,xeffs[i],xps[i],ndraws,bw_method) for i in range(len(xeffs))])
    joint_p_chi_p_chi_eff = p_chi_eff*p_chi_p_given_chi_eff

    return joint_p_chi_p_chi_eff

def chi_p_prior_given_chi_eff_q(q,aMax,xeff,xp,ndraws=10000,bw_method='scott'):

    """
    Function to calculate the conditional prior p(xp|xeff,q) on a single {xp,xeff,q} posterior sample.
    Called by `joint_prior_from_isotropic_spins`.

    INPUTS
    q: Single posterior mass ratio sample
    aMax: Maximimum spin magnitude considered
    xeff: Single effective inspiral spin sample
    xp: Single effective precessing spin value
    ndraws: Number of draws from the component spin priors used in numerically building interpolant

    RETURNS
    p_chi_p: Prior on xp, conditioned on given effective inspiral spin and mass ratio
    """

    # Draw random spin magnitudes.
    # Note that, given a fixed chi_eff, a1 can be no larger than (1+q)*chi_eff,
    # and a2 can be no larger than (1+q)*chi_eff/q
    a1 = np.random.random(ndraws)*aMax
    a2 = np.random.random(ndraws)*aMax

    # Draw random tilts for spin 2
    cost2 = 2.*np.random.random(ndraws)-1.

    # Finally, given our conditional value for chi_eff, we can solve for cost1
    # Note, though, that we still must require that the implied value of cost1 be *physical*
    cost1 = (xeff*(1.+q) - q*a2*cost2)/a1  

    # While any cost1 values remain unphysical, redraw a1, a2, and cost2, and recompute
    # Repeat as necessary
    while np.any(cost1<-1) or np.any(cost1>1):   
        to_replace = np.where((cost1<-1) | (cost1>1))[0]   
        a1[to_replace] = np.random.random(to_replace.size)*aMax
        a2[to_replace] = np.random.random(to_replace.size)*aMax
        cost2[to_replace] = 2.*np.random.random(to_replace.size)-1.    
        cost1 = (xeff*(1.+q) - q*a2*cost2)/a1   
            
    # Compute precessing spins and corresponding weights, build KDE
    # See `Joint-ChiEff-ChiP-Prior.ipynb` for a discussion of these weights
    Xp_draws = chi_p_from_components(a1,a2,cost1,cost2,q)
    jacobian_weights = (1.+q)/a1
    prior_kde = gaussian_kde(Xp_draws,weights=jacobian_weights,bw_method=bw_method)

    # Compute maximum chi_p
    if (1.+q)*np.abs(xeff)/q<aMax:
        max_Xp = aMax
    else:
        max_Xp = np.sqrt(aMax**2 - ((1.+q)*np.abs(xeff)-q)**2.)

    # Set up a grid slightly inside (0,max chi_p) and evaluate KDE
    reference_grid = np.linspace(0.05*max_Xp,0.95*max_Xp,50)
    reference_vals = prior_kde(reference_grid)

    # Manually prepend/append zeros at the boundaries
    reference_grid = np.concatenate([[0],reference_grid,[max_Xp]])
    reference_vals = np.concatenate([[0],reference_vals,[0]])
    norm_constant = np.trapz(reference_vals,reference_grid)

    # Interpolate!
    p_chi_p = np.interp(xp,reference_grid,reference_vals/norm_constant)
    return p_chi_p

def chi_p_from_components(a1,a2,cost1,cost2,q):

    """
    Helper function to define effective precessing spin parameter from component spins

    INPUTS
    a1: Primary dimensionless spin magnitude
    a2: Secondary's spin magnitude
    cost1: Cosine of the primary's spin-orbit tilt angle
    cost2: Cosine of the secondary's spin-orbit tilt
    q: Mass ratio

    RETRUNS
    chi_p: Corresponding precessing spin value
    """

    sint1 = np.sqrt(1.-cost1**2)
    sint2 = np.sqrt(1.-cost2**2)
    
    return np.maximum(a1*sint1,((3.+4.*q)/(4.+3.*q))*q*a2*sint2)
