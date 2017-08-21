""" 2D Ellipse fitting

    Fits an ellipse to a set of points (x_i, y_i) using the canonical
    representation:
        
     a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0 (1)
    
    Provided features
    -----------------
    
    The module provides several function related to ellipses:
    
    `fit_ellipse`:
        fits an ellipse from a set of points and return the parameters
        of the canonical representation (see above)
        
    `get_parameters`:
        converts canonical parameters into intuitive representation i.e.
        major and minor radii
    
    
    let a_ be the vector a_ = [a, b, c, d, e, f]'
    Let D be the (N x 6) design matrix:
        D = [z_1 z_2 ... z_n]'
        
        where 
        
        z_i = [ x_i^2, x_i * y_i, y_i^2, x_i, y_i, 1 ]'
    
    We want to minimize
    
        E = \sum_i (a_' * z_i)^2 = || D * a_ ||^2 = a_' * S * a
        
        where 
        
        S = D' * D

    If equation (1) corresponds to an ellipse we must have:
        
        4 * a * c - b^2 > 0
        
    Since equation (1) is unique up to a scaling factor we can impose:
        
        4 * a * c - b^2 = 1
        
    which can be written in matrix form as:
        
        a_' * C * a_ = 1
        
        with
        
        ::
        
            C = | 0 0  2 0 0 0 |
                | 0 -1 0 0 0 0 |
                | 2 0  0 0 0 0 |
                | 0 0  0 0 0 0 |
                | 0 0  0 0 0 0 |
                | 0 0  0 0 0 0 |

    So the problem reduces to:
        
        a_ = argmin a_' S a_
        s.t.                        (2)
            a_' * C * a_ = 1
        
    which is equivalent to solving:
        
        S a_ = l * C * a_           (3)
        
    where l is a Lagrange multiplier.
    
    Equation (2) is just a generalized eigen value problem. And the solution
    to (1) is the eigen vector corresponding to the smallest positive eigen
    value of (2).
    
    It can be prooved that (3) has 2 negative eigen values and one positive.
    The biggest eigen value the corresponds to the solution.
    
    Since C has negative eigen values solvers in scipy/numpy are not able
    to perform the eigen decomposition. To solve (2) we reduce the problem
    to a 3x3 eigen value problem for wich we can solve the problem
    analytically. For that, we split S and C into 3x3 blocks.    
    
    Let's define:
    
    ::
        
        S = | A  B |
            | B' E |
        
        C = | F  0 |
            | 0  0 |
            
        a_ = [x' y']'
        
    Rewritting (3) we get:
        
        A * x + B * y = l * F * x
        B'* x + E * y = 0
        
    which gives:
        
        y = - E^(-1) * B' * x
        (A - B * E^(-1) * B') * x = l * F * x (4)
    
    Equation (4) is a 3x3 eigen value problem that we can solve analytically.
    
    :Notes:

        Detailed explanations can be found in:
            
            *Direct Least square fitting of Ellipses*. A. Fitzgibbon, M. Pilu,
            and R. B. Fisher. Pattern Analysis and Machine Intelligence. 1999

    :Author: Alexis Mignon (c) 2012
    :E-mail: alexis.mignon@gmail.com
        
"""

import numpy as np
from scipy.linalg import inv, eigh, solve

def _find_max_eigval(S):
    """
    Finds the biggest generalized eigen value of the system
        
    S * x = l * C * x
        
    where
    
    ::
        
        C = | 0  0 2 |
            | 0 -1 0 |
            | 2  0 1 |

    Parameters:
    -----------    
    S : 3x3 matrix
    
    Returns:
    --------
    the highest eigen value
    """
    
    a = S[0,0]
    b = S[0,1]
    c = S[0,2]
    d = S[1,1]
    e = S[1,2]
    f = S[2,2]

    # computes the coefficients of the characteristic polynomial
    # det(S - x * C) = 0
    # Since the matrix is 3x3 we have a 3rd degree polynomial
    # _a * x**3 + _b * x**2 + _c * x + _d
    _a = -4
    _b = 4 * (c - d)
    _c = a * f - 4 * b * e + 4 * c * d - c * c
    _d = a * d * f - b * b * f - a * e * e + 2 * b * c * e - c  * c * d

    # computes the roots of the polynomial
    # there must be 2 negative roots and one
    # positive, i.e. the biggest one.
    x2, x1, x0 = sorted(np.roots([_a, _b, _c, _d] ))
    return x0

def _find_max_eigvec(S):
    """
    Computes the positive eigen value and the corresponding
    eigen vector of the system:
        
        S * x = l * C * x
    
        where
        ::
        
            C = | 0  0 2 |
                | 0 -1 0 |
                | 2  0 1 |
                
    Parameters:
    -----------    
    S : 3x3 matrix
    
    Returns:
    --------
        (l, u)
    
    l : float
        the positive eigen value
    
    u : the corresponding eigen vector
"""
     
    l = _find_max_eigval(S)

    a11 = S[0,0]
    a12 = S[0,1]
    a13 = S[0,2]
    a22 = S[1,1]
    a23 = S[1,2]
    
    u = np.array([
        a12 * a23 - (a13  - 2*l) * (a22 + l),
        a12 * (a13  - 2*l) - a23 * a11,
        a11 * (a22 + l) - a12 * a12
    ])

    c = 4 * u[0] * u[2] - u[1] * u[1]
    
    return l, u/np.sqrt(c)

def fit_ellipse(X):
    """ Fit an ellipse.
    
    Computes the best least squares parameters of an ellipse  expressed as:
        
        a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0
    
    Parameters
    ----------
    X : N x 2 array
        an array of N 2d points.
    
    Returns:
    --------
    an array containing the parameters:
        
        [ a , b, c, d, e, f]

"""
    x = X[:,0]
    y = X[:,1]

    # building the design matrix
    D = np.vstack([ x*x, x*y, y*y, x, y, np.ones(X.shape[0])]).T
    S = np.dot(D.T, D)
    
    S11 = S[:3][:,:3]
    S12 = S[:3][:,3:]
    S22 = S[3:][:,3:]
    
    S22_inv = inv(S22)
    S22_inv_S21 = np.dot(inv(S22), S12.T)
    
    Sc =  S11 - np.dot(S12, S22_inv_S21)
    l, a = _find_max_eigvec(Sc)
    
    b = - np.dot(S22_inv_S21, a)

    return np.hstack([a,b])

def create_ellipse(r, xc, alpha, n=100, angle_range=(0,2*np.pi)):
    """ Create points on an ellipse with uniform angle step
    
    Parameters
    ----------
    r: tuple
        (rx, ry): major an minor radii of the ellipse. Radii are supposed to
        be given in descending order. No check will be done.
    xc : tuple
        x and y coordinates of the center of the ellipse
    alpha : float
        angle between the x axis and the major axis of the ellipse
    n : int, optional
        The number of points to create
    angle_range : tuple (a0, a1)
        angles between which points are created.
        
    Returns
    -------
        (n * 2) array of points 
"""
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])
    
    a0,a1 = angle_range
    angles = np.linspace(a0,a1,n)
    X = np.vstack([ np.cos(angles) * r[0], np.sin(angles) * r[1]]).T
    return np.dot(X,R.T) + xc

def create_cassini_oval(r, xc, alpha, n=100, angle_range=(0,2*np.pi)):
    """ Create points on an Cassini oval with uniform angle step
    reference: http://virtualmathmuseum.org/Curves/cassinian_oval/Cassinian_Oval.pdf
    
    Parameters
    ----------
    r: tuple
        (rx, ry): major an minor radii of the ellipse. Radii are supposed to
        be given in descending order. No check will be done.
    xc : tuple
        x and y coordinates of the center of the ellipse
    alpha : float
        angle between the x axis and the major axis of the ellipse
    n : int, optional
        The number of points to create
    angle_range : tuple (a0, a1)
        angles between which points are created.
        
    Returns
    -------
        (n * 2) array of points 
"""
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])
    a0,a1 = angle_range
    angles = np.linspace(a0,a1,n)
    a = np.sqrt((r[0]**2-r[1]**2)/2)
    b = np.sqrt((r[0]**2+r[1]**2)/2)
    M = 2*a**2*np.cos(2*angles)+2*np.sqrt((-a**4+b**4)+a**4*np.cos(2*angles)**2)
    X = np.vstack([ np.cos(angles) *np.sqrt(M/2), np.sin(angles) * np.sqrt(M/2)]).T
#     x = np.cos(angles)*np.sqrt(M/2) + xc[0]
#     y = np.sin(angles)*np.sqrt(M/2) + xc[1]
#     points = np.array([[x[i],y[i]] for i in range(angles.size)])
    return np.dot(X,R.T) + xc

def get_parameters(x):
    """
    Computes 'natural' parameters of an ellipse given the parameters
    of the canonical equation:
        
        a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0
    
    Parameters:
    -----------
    x : array_like
        An array of 6 elements corresponding to the coefficients of the
        canonical equation (see above)
    
    Returns:
    --------
        tuple (rx, ry), (xc, yc), alpha
        
    (rx, ry) : tuple
        Radii of the major and minor axes
    
    (xc, yc) : tuple
        coordinates of the center
    
    alpha : float
        angle between the x axis and the major axis
    
    :Note:

        Computed the parameters of the ellipse when it is expressed as:
            
            x'^2/rx^2 + y'/ry^2 = 1
            
        where x' and y' correpsond to the rotated coordinates:    
            
            x' =  cos(alpha)(x-xc) + sin(alpha)(y-yc)
            y' = -sin(alpha)(x-xc) + cos(alpha)(y-yc)
        
        Which can be put in matrix form as
        
            (X-Xc)' R D R' (X-Xc) = 1
        
        where
        ::
            
              X = [x y] and Xc = [xc yc]
              
              R = [ cos(alpha) -sin(alpha)]
                  [ sin(alpha) cos(alpha) ]
                  
              D = [ 1/rx^2           0    ]
                  [    0          1/ry^2  ]
                    
        Parameters are given as the parameter of the conic:
            
            a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0
        
        In matrix form we have:
            
            X' A X + B' X + f = 0
            
            where
            ::
                  X = [ x  y ]'
            
                  A = [ a b/2]
                      [b/2 c ]
                      
                  B = [ d  e ]'
              
        Any ellipse can be written as:
            
                (X - Xc)' A (X - Xc)  = r^2
                
        which develops in:
            
            X'A X - 2 * Xc' A X + Xc' A Xc - r^2 = 0
            
            So we have:
                
                B = - 2 * A Xc
                
            and
            
                f = Xc' A Xc - r^2
                
            and thus:
                
                Xc = -1/2 * A^(-1) B
                r^2 = Xc' A Xc - f
            
            We also see that
            
            1/r^2 * (X - Xc)' A (X - Xc) = (X-Xc)' R D R' (X-Xc) = 1
            
            By performing eigen decomposition on A = U L U', we obtain
            
                R = U
                
                and 
                
                lx / r^2 = 1/rx^2
                ly / r^2 = 1/ry^2
                
            hence
            
                rx^2 = r^2 / lx
                ry^2 = r^2 / ly
                
            the angle alpha is finally determined using
            ::
                
                U = | u11,  u12 | = | cos(alpha)  sin(alpha)|
                    |-u12,  u22 |   | -sin(alpha) cos(alpha)|
                alpha = sign(u12) * arccos(u11)
    """
    a,b,c,d,e,f = x
    
    A = np.array([
        [ a, b/2 ],
        [b/2, c  ]
    ])    
    
    B = np.array([d,e])    
    
    w,u = eigh(A)
    
    Xc = solve(-2*A,B)
    r2 = -0.5 * np.inner(Xc,B) - f
    
    rr2 = r2 / w
    
    alpha = np.arccos(u[0,0])
    if alpha > np.pi/2:
        alpha = alpha - np.pi
        
    alpha *= np.sign(u[0,1])
    
    return tuple(np.sqrt(rr2)), tuple(Xc), alpha