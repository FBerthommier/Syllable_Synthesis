import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks 

def softrect(x, S):
    return (np.sqrt(x**2 + S) + x) / 2

def spectrelec(w, A, zr, l, no_vibration, CONST_DAT):
    c = CONST_DAT[0]
    ro = CONST_DAT[1]
    lambda_ = CONST_DAT[2]
    eta = CONST_DAT[3]
    mu = CONST_DAT[4]
    cp = CONST_DAT[5]
    bp = CONST_DAT[6]
    mp = CONST_DAT[7]

    S = 2 * np.sqrt(A * np.pi)
    L = ro / A * l
    C = A * l / (ro * c * c)

    R_coef = np.sqrt(ro * mu / (2 * w))
    G_coef = (eta - 1) / (ro * c**2) * np.sqrt(lambda_ * w / (2 * cp * ro))
    R = S * l / (A**2) * R_coef
    G = S * l * G_coef
    

    YP_coef = 1 / (bp**2 + mp**2 * w**2)
    YP = S * l * ((bp - 1j * mp * w) * YP_coef)

    if no_vibration == 1:
        YP = 0

    Z = R + 1j * L * w
    Y = G + 1j * C * w + YP
    aa = (1 + (Z * Y / 2))
    
    bb = - (Z + Z**2 * Y / 4)
    cc = - Y
    dd = aa
    aaa = aa[0, :]  
    bbb = bb[0, :] 
    ccc = cc[0, :] 
    ddd = dd[0, :]  
    
    for ind in range(len(A) - 1):      
        proda = aa[ind + 1, :] * aaa + bb[ind + 1, :] * ccc
        prodb = aa[ind + 1, :] * bbb + bb[ind + 1, :] * ddd
        prodc = cc[ind + 1, :] * aaa + dd[ind + 1, :] * ccc
        prodd = cc[ind + 1, :] * bbb + dd[ind + 1, :] * ddd
        aaa = proda
        bbb = prodb
        ccc = prodc
        ddd = prodd
        H = np.ones_like(aaa) / (aaa - ccc * zr)
    return H

# %---c----C----c----C----c----C----c----C----c----C----c----C----c----C
# %
# %                     INSTITUT DE LA COMMUNICATION PARLEE
# %                               GRENOBLE, FRANCE
# %
# %                               ** nraph3.ftn **
# %
# %       Creation : 7 Juin 1987
# %
# %
# % Hugo SANCHEZ
# % Creation : 19 Juin 1984
# % Mis a jour : 19 Juin 1984
# % Derniere modification : Pierre BADIN (21 novembre 1985)
# %     "         "       : 86-08-29 (Possibilite de recherche de zeros
# %                       et de poles simples et complexes conjugues,
# %                       aussi bien dans TP que dans TZ.
# %     "         "       : 87-02-13 (Commentaires pour une version globale 0)
# % Derniere Modification : 87-03-04 (Passage sur VAX)
# %     "         "       : 87-06-08 (Conditions sur INAS)
# %-----------------------------------------------------------------------------
# %       Entree
# %               FE, BNPE, Estimation de depart des parametres
# %               ITERMX, Nombre d'iterations autorisees
# %               MODE =1 recherche des poles de TP
# %                    =2     "         poles de TZ
# %                    =3     "     des zeros de TP
# %                    =4     "         zeros de TZ
# %               FMAX, Frequence maximum autorisee
# %
# %       Sorties
# %               F, BP, Parametres estimes par le programme
# %               NRAPH, Nombre d'iterations (=0 si pas de solution)

def nraph_oral(FE, BNPE, ITERMX, FMAX, area, F_form, NF, no_vibration, CONST_DAT):
    DELTAS = complex(30, 30)
    SEUIL = 0.3
    ITER = 1
    SI = complex(-BNPE * np.pi, FE.item() * 2 * np.pi)

    F = np.nan
    BP = np.nan
    while ITER < ITERMX:
        FIcx = -1j * SI / (2 * np.pi)
        Q = 1 / aire2spectre_cor_oral(area, 1, FIcx, FIcx, F_form,NF, no_vibration, CONST_DAT)
        SIP = SI + DELTAS
        FIPcx = -1j * SIP / (2 * np.pi)
        QP = 1 / aire2spectre_cor_oral(area, 1, FIPcx, FIPcx, F_form,NF, no_vibration, CONST_DAT)
        Q1D = (QP - Q) / DELTAS
        SISU = SI - (Q / Q1D)
        if abs(SISU - SI) < SEUIL:
            F = np.imag(SISU) / (2 * np.pi)
            BP = -np.real(SISU) / np.pi
            return F, BP
        else:
            SI = SISU
            ITER += 1
    
    return np.float64(F), np.float64(BP)

# % function [H_oral, Y_oral] = aire2spectre_oral(ORAL{, nbfreq, Fmax, Fmin});
# %
# % Calcule:
# %   - la FT totale pour les freq Fmin à Fmax à partir de la fonction d'aire ORAL et NASAL
# %   - Les formants associés
# %   - la FT orale
# %
# % Entrées
# %      ORAL.A(nb_tubes_oral)	  : Aires des tubes oraux (glotte -> lèvres)
# %      ORAL.L(nb_tubes_oral)	  : Longueurs des tubes oraux (glotte -> lèvres)
# %
# %      nbfreq ([256])           : Nombre de point d'échantillonnage fréquentiel entre 0 et Fmax
# %      Fmax ([5000])            : Fréquence maximale de la fonction de transfert
# %      Fmin ([Fmax/nbfreq])     : Fréquence minimale de la fonction de transfert
# %
# %
# % Sorties
# %      H_oral(nbfreq)           : Fonction de transfert orale (linéaire)
# %      Y_oral(nbfreq)           : Impédance d'entrée du tuyau oral

def aire2spectre_oral(area, nbfreq, Fmax, Fmin, no_vibration, CONST_DAT):
    c = CONST_DAT[0]
    rho = CONST_DAT[1]
    f = np.linspace(Fmin, Fmax, nbfreq)
    w = np.squeeze(2 * np.pi * f)
    
    Zr_oral = rho / (2 * np.pi * c) * (w ** 2) + 1j * 8 * rho / (3 * np.pi * np.sqrt(np.pi * area[-1,1])) * w
    H_oral = spectrelec(w, area[:, 1].reshape(-1,1), Zr_oral, area[:, 0].reshape(-1,1), no_vibration, CONST_DAT)  # 0 = vibration des parois
    return H_oral

def aire2spectre_cor_oral(area, nbfreq, Fmax, Fmin, F_form, NF, no_vibration, CONST_DAT):
    H_eq = aire2spectre_oral(area, nbfreq, Fmax, Fmin, no_vibration, CONST_DAT)

    f = np.linspace(Fmin, Fmax, nbfreq)
    if np.isrealobj(f):
        SI = 1j * 2 * np.pi * f
    else:
        SI = 2 * np.pi * 1j * f

    for I in range(NF):
        SI1 = complex(F_form[I, 2] * np.pi, F_form[I, 1] * 2 * np.pi)
        H_eq *= np.squeeze(((SI - SI1) * (SI - np.conj(SI1))) / (SI1 * np.conj(SI1)))

    return H_eq

def vtn2frm_ftr_oral(area, nbfreq, Fmax, Fmin, no_vibration,CONST_DAT):
    f = np.linspace(Fmin, Fmax, nbfreq)

    FE = 150
    BNPE = 50
    FINC = 100
    ITERMX = 100
    seuil2 = 10
    FMAX = Fmax

    F_form = np.zeros((100, 3)) 

    NF = 0
    F = 0
    while F <= FMAX:
        if NF > 0:
            FE = F + FINC
        H_eq = aire2spectre_cor_oral(area, nbfreq, Fmax, Fmin, F_form, NF, no_vibration, CONST_DAT)
        ind_maxi, _ = find_peaks(np.asarray(20 * np.log10(np.abs(H_eq)), dtype=np.float64).flatten())
        ind_mini, _ = find_peaks(np.asarray(-20 * np.log10(np.abs(H_eq)), dtype=np.float64).flatten())
        frq_min = f[ind_mini]
        frq_max = f[ind_maxi]
        FEST = frq_max
        if any(FEST): 
            FE = FEST[0]
        
        # Recherche des poles de H_eq par nraph
        F, BP = nraph_oral(FE, BNPE, ITERMX, FMAX, area, F_form, NF, no_vibration, CONST_DAT)
        F_form[NF,:] = [0, F.item(), BP.item()] 
        NF += 1

    H_eq = aire2spectre_oral(area, nbfreq, Fmax, Fmin, no_vibration, CONST_DAT)

    nbformants = 3
    F_form1 = np.zeros((nbformants, 1))
    for k in range(min(nbformants, NF)):
        F_form1[k] = F_form[k, 1]

    return H_eq, F_form1

def showgui(gui):
    fig, ax = plt.subplots()
    # Constantes utilisées pour l'ajustement
    mx = -0.0153  
    my = 138.4566
    
    # Extraction des données sagittales et mise à l'échelle
    sag = gui['sagittal'] / 29.5 

    # Tracé des données sur l'axe `ax`
    h1, = ax.plot(
        (sag[:, 0] - sag[55, 0] + mx),  # Ajustement des coordonnées x
        (sag[:, 1] + sag[55, 1]) - my,  # Ajustement des coordonnées y
        '-b'  # Style de ligne bleu
    )
    
    # Modification des propriétés de la ligne
    h1.set_color('r')  # Changer la couleur de la ligne en vert
    h1.set_linewidth(2)  # Épaisseur de la ligne

    # Définition des limites de l'axe et de l'aspect
    ax.axis([ -1.5, 8.5, -7.5, 2.5])
    ax.set_aspect('equal', adjustable='box')
    
    # Suppression des axes
    ax.axis('off')
    
    return fig  # Retourne le pointeur sur l'objet tracé

###################################################################################
# % le modele VLAM a ete ecrit par Shinji MAEDA
# % le modele VLAM transcrit du C en Matlab par David POCHIC et Nassim ZGA modifs JLS - LJB (palais & pharynx) et modif pour generer des uvulaires
# % VLAM modification avec rotation du conduit
# % parametres d'entree : gui.prm(1:10)
# % parametres de sortie : gui.sagittal gui.area
# % Modifie Ete 2006 Par David POCHIC et Nassim ZGA
# % /* VARIATION SELON L'AGE */
###################################################################################

def initVLAMLength(Ap, L):
    gui = {}
    gui['sagittal'] = np.zeros((58, 2))
    gui['inci'] = np.zeros((1, 2))
    gui['area'] = np.zeros((29, 2))
    gui['PD'] = 0
    gui['LH'] = 0
    gui['LHI'] = 0
    gui['B'] = 0
    gui['A'] = 0

    kage = 5.4749 * L - 407.1374
    gui['prm'] = np.zeros(11)
    gui['prm'][0:7] = Ap
    gui['prm'][7] = kage
    gui['prm'][8] = 0
    gui['prm'][10] = 0
    gui['prm'][9] = 0
    return gui

def vlam2009NN(gui):
    
    k_age = gui['prm'][7]  # 650 = adulte male, 0 = nouveau ne
    k_age_max = 650

    parametres = np.zeros(10)
    parametres[0] = gui['prm'][0]  # Jaw
    parametres[1] = gui['prm'][1]  # Body
    parametres[2] = gui['prm'][2]  # Drsm    
    parametres[3] = gui['prm'][3]  # Apex
    parametres[4] = gui['prm'][5]  # LipP
    parametres[5] = gui['prm'][4]  # LipH   
    parametres[6] = gui['prm'][6]  # Larynx
    parametres[7] = gui['prm'][7]  # Age
    
    # mode Human
    parametres[8] = 0.8  # boucles;   % k_Pharynx
    parametres[9] = 0.35  # boucles1;  % k_Mouth
    
    # parametres d'applatissement et de rotation
    Flat_P = -gui['prm'][8]
    Flat_T = -gui['prm'][10]
    phi = gui['prm'][9]
    AF_correc = 1
    
    A_tng = np.array([[1.000000, 0.000000, 0.000000, 0.000000],
                      [-0.464047, 0.098776, -0.251690, 0.000000],
                      [-0.328015, 0.337579, -0.283667, 0.000000],
                      [-0.213039, 0.485565, -0.283533, 0.000000],
                      [-0.302565, 0.705432, -0.379044, 0.000000],
                      [-0.327806, 0.786897, -0.388116, 0.000000],
                      [-0.325065, 0.852409, -0.285125, 0.000000],
                      [-0.325739, 0.904725, -0.142602, 0.000000],
                      [-0.313741, 0.926339, 0.021042, 0.000000],
                      [-0.288138, 0.924019, 0.131949, 0.000000],
                      [-0.249008, 0.909585, 0.250320, 0.000000],
                      [-0.196936, 0.882236, 0.369083, 0.000000],
                      [-0.128884, 0.830243, 0.499894, 0.000000],
                      [-0.040825, 0.730520, 0.651662, 0.112048],
                      [0.073420, 0.543080, 0.807947, 0.126204],
                      [0.202726, 0.230555, 0.919065, 0.163735],
                      [0.298853, -0.162541, 0.899074, 0.213884],
                      [0.332785, -0.491647, 0.748869, 0.243163],
                      [0.349955, -0.681313, 0.567615, 0.245295],
                      [0.377277, -0.771200, 0.410502, 0.249425],
                      [0.422713, -0.804874, 0.270513, 0.274015],
                      [0.474635, -0.797704, 0.129324, 0.314454],
                      [0.526087, -0.746938, -0.026201, 0.366149],
                      [0.549466, -0.643572, -0.190005, 0.422848],
                      [0.494200, -0.504012, -0.350434, 0.488056],
                      [0.448797, -0.417352, -0.445410, 0.500909]])

    s_tng = np.array([27.674635, 29.947931, 44.694466, 99.310226, 96.871323,
                      84.140404, 78.357513, 73.387718, 72.926758, 71.453232,
                      69.288765, 66.615509, 63.603722, 59.964859, 56.695446,
                      56.415058, 62.016468, 73.235176, 84.008438, 91.488312,
                      94.124176, 95.246323, 93.516365, 93.000343, 100.934669,
                      106.512482])

    u_tng = np.array([104.271675, 443.988434, 450.481689, 399.942200, 348.603088,
                      351.181122, 365.404633, 370.290955, 356.202301, 341.890167,
                      332.117523, 326.826599, 326.512512, 331.631989, 343.175323,
                      361.265900, 385.231201, 411.826599, 435.691711, 455.040466,
                      462.736023, 453.025055, 432.250488, 407.358368, 384.551056,
                      363.836212])

    A_lip = np.array([[1.000000, 0.000000, 0.000000],
                      [0.178244, -0.395733, 0.888897],
                      [-0.154638, 0.987971, 0.000000],
                      [-0.217332, 0.825187, -0.303429]])

    s_lip = np.array([27.674635, 33.068081, 99.392258, 213.996170])
    u_lip = np.array([104.271675, 122.812141, 135.938339, 460.440857])

    A_lrx = np.array([[1.000000, 0.000000],
                  [-0.208338, 0.262446],
                  [0.127814, 0.991798],
                  [-0.131840, 0.300784],
                  [0.097688, 0.934267]])

    s_lrx = np.array([27.674635,
                  41.593315,
                  65.562340,
                  44.372742,
                  66.147499])

    u_lrx = np.array([104.271675,
                  143.138733,
                  -948.229309,
                  404.678223,
                  -962.936401])

    u_wal = np.array([550.196533,
                  604.878601,
                  674.127197,
                  678.776489,
                  665.905579,
                  653.312134,
                  643.223511,
                  633.836243,
                  636.994202,
                  668.834290,
                  703.098267,
                  600,
                  610,
                  610,
                  605,
                  600,
                  600,
                  600,
                  600,
                  600,
                  600,
                  600,
                  600,
                  600,
                  600])

    ix0 = 3000
    iy0 = 1850

    vp_map = 1 / 29.5
    TEKvt = 188.679245
    TEKvt *= vp_map

    inci_x = 2212.354492
    inci_y = 1999.574219
    inci_lip = 0.8

    inci_x = (inci_x - ix0) / TEKvt
    inci_y = (inci_y - iy0) / TEKvt
    inci_lip_vp = inci_lip / vp_map

    s_tng /= TEKvt
    u_tng /= TEKvt
    s_lip /= TEKvt
    u_lip /= TEKvt
    s_lrx /= TEKvt
    u_lrx /= TEKvt
    u_wal /= TEKvt

    ix0 = 2200
    iy0 = 2000

    Pharynx_scale = k_age * (parametres[8]) / k_age_max + 0.3
    Mouth_scale = k_age * (parametres[9]) / k_age_max + 0.65

    r = 5
    dl = 0.5
    m1 = 14
    m2 = 11
    m3 = 6
    omega = -11.25
    theta = 11.25

    r_vp = r / vp_map
    dlPharynx_vp = Pharynx_scale * dl / vp_map
    dlPalatal_vp = Mouth_scale * dl / vp_map
    ome = np.pi * omega / 180.
    the = np.pi * theta / 180.

    dx_i = dlPharynx_vp * np.cos(ome - np.pi / 2.)
    dy_i = dlPharynx_vp * np.sin(ome - np.pi / 2.)
    dx_e = r_vp * np.cos(ome)
    dy_e = r_vp * np.sin(ome)

    igd1 = np.array([(dx_i * np.arange(m1 - 1, -1, -1)) + ix0,
                  (dy_i * np.arange(m1 - 1, -1, -1)) + iy0]).T

    egd1 = np.array([dx_e + igd1[:, 0],
                 dy_e + igd1[:, 1]]).T

    gam = the * np.arange(1, m2 + 1) + ome
    igd2 = np.array([ix0 * np.ones(11),
                 iy0 * np.ones(11)]).T
    egd2 = np.array([r_vp * np.cos(gam) + ix0,
                 r_vp * np.sin(gam) + iy0]).T

    dx_i = dlPalatal_vp * np.cos(gam[-1] + np.pi / 2.)
    dy_i = dlPalatal_vp * np.sin(gam[-1] + np.pi / 2.)
    dx_e = r_vp * np.cos(gam[-1])
    dy_e = r_vp * np.sin(gam[-1])

    igd3 = np.array([(dx_i * np.arange(1, m3 + 1)) + ix0,
                  (dy_i * np.arange(1, m3 + 1)) + iy0]).T
    egd3 = np.array([dx_e + igd3[:, 0],
                 dy_e + igd3[:, 1]]).T

    igd = np.vstack((igd1, igd2, igd3))
    egd = np.vstack((egd1, egd2, egd3))

    p = egd[:, 0] - igd[:, 0]
    q = egd[:, 1] - igd[:, 1]
    s = np.sqrt(p * p + q * q)
    vtos = np.vstack((p / s, q / s)).T

    omega = -11.25000
    Pharynx_scale = k_age * (parametres[8]) / k_age_max + 0.3
    JAW = 0
    LIP = 1
    TNG = 2
    ix0 = 2200
    iy0 = 2000
    NP = 29

    v = A_tng @ parametres[:4]
    v_tng = s_tng * v + u_tng
    v = A_lip @ np.array([parametres[0], parametres[4], parametres[5]])
    v_lip = s_lip * v + u_lip

    i = np.where(v_lip < 0)
    v_lip[i] = 0
    v = A_lrx @ np.array([parametres[0], parametres[6]])
    v_lrx = s_lrx * v + u_lrx

    omg = np.pi * omega / 180.

    x1 = v_lrx[JAW + 1]
    y1 = v_lrx[JAW + 2]
    b = y1 - np.tan(omg + np.pi / 2) * x1
    x0 = b / (np.tan(omg) - np.tan(omg + np.pi / 2))
    y0 = np.tan(omg) * x0

    x1 = Pharynx_scale * (x1 - x0) + x0
    y1 = Pharynx_scale * (y1 - y0) + y0
    b = y1 - np.tan(omg) * x1
    x0 = b / (np.tan(omg + np.pi / 2) - np.tan(omg))
    y0 = np.tan(omg + np.pi / 2) * x0

    ivt1 = np.array([[Pharynx_scale * (x1 - x0) + x0 + ix0,
                   Pharynx_scale * (y1 - y0) + y0 + iy0]])

    x1 = v_lrx[JAW + 3]
    y1 = v_lrx[JAW + 4]
    b = y1 - np.tan(omg + np.pi / 2) * x1
    x0 = b / (np.tan(omg) - np.tan(omg + np.pi / 2))
    y0 = np.tan(omg) * x0

    x1 = Pharynx_scale * (x1 - x0) + x0
    y1 = Pharynx_scale * (y1 - y0) + y0
    b = y1 - np.tan(omg) * x1
    x0 = b / (np.tan(omg + np.pi / 2) - np.tan(omg))
    y0 = np.tan(omg + np.pi / 2) * x0

    evt1 = np.array([[Pharynx_scale * (x1 - x0) + x0 + ix0,
                   Pharynx_scale * (y1 - y0) + y0 + iy0]])

    # Scale factors
    scale_factor1 = Pharynx_scale * np.ones((8, 1))
    scale_factor2 = (Mouth_scale - Pharynx_scale) * np.arange(1, m2 + 1).reshape(-1, 1) / m2 + Pharynx_scale
    scale_factor3 = Mouth_scale * np.ones((6, 1))

    # Combine the scale factors
    scale_factor = np.vstack((scale_factor1, scale_factor2, scale_factor3))

    # Block tongue contour at walls
    v = scale_factor.flatten() * np.minimum(v_tng[1:26], u_wal)

    # Inside tongue contour
    xy1 = np.column_stack((vtos[6:31, 0] * v, vtos[6:31, 1] * v)) + igd[6:31, :]

    # Temporary variable for tongue contour at walls
    vtos_wal = np.column_stack((vtos[6:31, 0] * u_wal, vtos[6:31, 1] * u_wal))

    # Outside tongue contour
    xy2 = np.column_stack((vtos_wal[:, 0] * scale_factor.flatten(), vtos_wal[:, 1] * scale_factor.flatten())) + igd[6:31, :]

    # Intermediate variables
    ivt2 = (ivt1 + xy1[0, :]) / 2
    ivt3 = xy1

    evt2 = (evt1 + xy2[0, :]) / 2
    evt3 = xy2
    ivt4 = np.vstack((ivt1, ivt2, ivt3))
    evt4 = np.vstack((evt1, evt2, evt3))

    omg = np.pi * theta / 180.

    x1 = inci_x
    y1 = inci_y + inci_lip_vp
    b = y1 - np.tan(omg + np.pi / 2) * x1
    x0 = b / (np.tan(omg) - np.tan(omg + np.pi / 2))
    y0 = np.tan(omg) * x0

    x1 = Mouth_scale * (x1 - x0) + x0
    y1 = Mouth_scale * (y1 - y0) + y0
    b = y1 - np.tan(omg) * x1
    x0 = b / (np.tan(omg + np.pi / 2) - np.tan(omg))
    y0 = np.tan(omg + np.pi / 2) * x0
    evtn = np.zeros((2, 2))
    evtn[0, 0] = Mouth_scale * (x1 - x0) + x0 + ix0
    evtn[0, 1] = Mouth_scale * (y1 - y0) + y0 + iy0
    ivtn = np.zeros((2, 2))
    ivtn[0, 0] = evtn[0, 0]
    ivtn[0, 1] = evtn[0, 1] - Mouth_scale * v_lip[2]

    evtn[1, 0] = evtn[0, 0] - Mouth_scale * v_lip[1]
    evtn[1, 1] = evtn[0, 1]
    ivtn[1, 0] = evtn[1, 0]
    ivtn[1, 1] = ivtn[0, 1]

    ivtn0 = ivtn - np.array([ix0, iy0])
    evtn0 = evtn - np.array([ix0, iy0])

    omg0 = 11.5 * np.pi / 180

    lip_h = Mouth_scale * v_lip[2] / 2
    lip_w = Mouth_scale * v_lip[3] / 2

    ivt = np.vstack((ivt4, ivtn))
    evt = np.vstack((evt4, evtn))
    sagittal = np.vstack((np.flipud(ivt), evt))

    x56 = sagittal[55, 0]
    y56 = sagittal[55, 1]
    
    x42 = sagittal[41, 0]
    y42 = sagittal[41, 1]

    x30 = sagittal[29, 0]
    y30 = sagittal[29, 1]

    x29 = sagittal[28, 0]
    y29 = sagittal[28, 1]

    x29_30 = (x29 + x30) / 2
    y29_30 = (y29 + y30) / 2

    gui['PD'] = np.sqrt((x42 - x56) ** 2 + (y42 - y56) ** 2)
    gui['LH'] = np.sqrt((x29_30 - x42) ** 2 + (y29_30 - y42) ** 2)
    gui['LHI'] = gui['LH'] / gui['PD']

    k = 3 / 5  # Facteur 0< k <1
    # 1=>pharynx plat
    # 0=>aucune retouche

    x36 = sagittal[35, 0]
    y36 = sagittal[35, 1]

    x34 = sagittal[33, 0]
    y34 = sagittal[33, 1]
    ab = np.linalg.solve(np.array([[x42, 1], [x34, 1]]), np.array([y42, y34]))
    
    sagint = sagittal[:33, :]
    sagmiddle = sagittal[33:41, :]
    sagext = sagittal[41:, :]

    alpha = -np.arctan((sagittal[41, 1] - sagittal[33, 1]) / (sagittal[41, 0] - sagittal[33, 0]))
    centre = np.array([0, ab[1]])
    
    # Premiere rotation
    sag_p = sagmiddle
    sag_p0 = sag_p - centre
    

    sag_p = (np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]).dot(sag_p0.T)).T
    sag_p = np.array([sag_p[:, 0] + centre[0], sag_p[:, 1] + centre[1]]).T

    b = ab[1] * np.ones((8, 1))
    sag_p = np.hstack((sag_p[:, 0].reshape(-1, 1), b))

    # Deuxieme rotation en sens inverse
    alpha = -alpha
    sag_p0 = np.array([sag_p[:, 0] - centre[0], sag_p[:, 1] - centre[1]]).T
    sag_p = (np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]).dot(sag_p0.T)).T
    sag_p = np.array([sag_p[:, 0] + centre[0], sag_p[:, 1] + centre[1]]).T

    mat_vect = np.array([sag_p[:, 0] - sagmiddle[:, 0], sag_p[:, 1] - sagmiddle[:, 1]]).T
    sagmiddle = sagmiddle + k * mat_vect
    sagittal = np.vstack((sagint, sagmiddle, sagext))  # Sagittal final

    if Flat_P != 0 or Flat_T != 0:
        x50 = sagittal[49, 0]
        y50 = sagittal[49, 1]
    
        x9 = sagittal[8, 0]
        y9 = sagittal[8, 1]
    
        x44 = sagittal[43, 0]
        y44 = sagittal[43, 1]
    
        x15 = sagittal[14, 0]
        y15 = sagittal[14, 1]

        Y1 = np.array([y44, y15])
        X1 = np.array([[x44, 1], [x15, 1]])
        A1 = np.linalg.lstsq(X1, Y1, rcond=None)[0]

        Y2 = np.array([y50, y9])
        X2 = np.array([[x50, 1], [x9, 1]])

        A2 = np.linalg.lstsq(X2, Y2, rcond=None)[0]

        A = np.array([[-A1[0], 1], [-A2[0], 1]])
        B = np.array([[A1[1]], [A2[1]]])

        intersec = np.linalg.lstsq(A, B, rcond=None)[0]

        if Flat_P != 0:
            eps = np.arange(0, np.pi, np.pi/14)
            sagpalais = sagittal[41:56, :]

            vect_AI = np.column_stack((intersec[0] - sagpalais[:, 0], intersec[1] - sagpalais[:, 1]))

            sagpalais += Flat_P * np.column_stack((np.sin(eps) * vect_AI[:, 0], np.sin(eps) * vect_AI[:, 1]))
            sagittal = np.vstack((sagittal[:41, :], sagpalais, sagittal[56:, :]))

        if Flat_T != 0:
            eps = np.arange(0, np.pi, np.pi/15)
            saglangue = sagittal[2:18, :]

            vect_AI = np.column_stack((intersec[0] - saglangue[:, 0], intersec[1] - saglangue[:, 1]))

            saglangue += Flat_T * np.column_stack((np.sin(eps) * vect_AI[:, 0], np.sin(eps) * vect_AI[:, 1]))
            sagittal = np.vstack((sagittal[:2, :], saglangue, sagittal[18:, :]))

    if phi != 0:
        phi = phi * np.pi / 180.0
        supp = np.array([50, 50])
        cphi = np.array([x42 + supp[0], y42 + supp[1]])

        sagintrot = sagittal[:17, :]
        sagmiddle = sagittal[17:41, :]
        sagextrot = sagittal[41:, :]

        sagintrot0 = sagintrot - cphi
        sagextrot0 = sagextrot - cphi

        sagintrot = (np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]).dot(sagintrot0.T)).T
        sagextrot = (np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]).dot(sagextrot0.T)).T

        sagintrot += cphi
        sagextrot += cphi

        sagittal = np.vstack((sagintrot, sagmiddle, sagextrot))

    gui['inci'] = sagittal[55, :]  # On sauvegarde la position de l'incisive
    gui['sagittal'] = sagittal


    alpha = np.array([  1.80000,
                    1.80000,
                    1.80000,
                    1.80000,
                    1.80000,
                    1.80000,
                    1.80000,
                    1.80000,
                    1.80000,
                    1.80000,
                    1.80000,
                    1.80000,
                    1.80000,
                    1.70000,
                    1.70000,
                    1.70000,
                    1.70000,
                    1.70000,
                    1.70000,
                    1.70000,
                    1.70000,
                    1.70000,
                    1.80000,
                    1.80000,
                    1.90000,
                    2.00000,
                    2.60000])
                
    beta = np.array([1.20000,
                1.20000,
                1.20000,
                1.20000,
                1.20000,
                1.20000,
                1.20000,
                1.20000,
                1.20000,
                1.20000,
                1.20000, 
                1.20000,
                1.30000,
                1.40000,
                1.40000,
                1.50000,
                1.50000,
                1.50000,
                1.50000,
                1.50000,
                1.50000,
                1.50000,
                1.50000,
                1.50000,
                1.50000,
                1.50000,
                1.50000])        
    
    c = AF_correc * vp_map
    cc = c * c

    nb_permut = len(ivt) - 1
    # Indices à revoir
    ivt = np.flipud(sagittal[0:29, :])
    evt = sagittal[29:58, :]

    ivt2 = np.roll(ivt, nb_permut, axis=0)
    evt2 = np.roll(evt, nb_permut, axis=0)
    
    p = np.sqrt((ivt[:, 0] - ivt2[:, 0])**2 + (ivt[:, 1] - ivt2[:, 1])**2)
    q = np.sqrt((evt[:, 0] - evt2[:, 0])**2 + (evt[:, 1] - evt2[:, 1])**2)
    r = np.sqrt((evt[:, 0] - ivt[:, 0])**2 + (evt[:, 1] - ivt[:, 1])**2)
    s = np.sqrt((ivt2[:, 0] - evt2[:, 0])**2 + (ivt2[:, 1] - evt2[:, 1])**2)
    t = np.sqrt((ivt[:, 0] - evt2[:, 0])**2 + (ivt[:, 1] - evt2[:, 1])**2)

    p = p[:27]
    q = q[:27]
    r = r[:27]
    s = s[:27]
    t = t[:27]

    a1 = 0.5 * (p + s + t)
    a2 = 0.5 * (q + r + t)
    s1 = np.sqrt(a1 * (a1 - p) * (a1 - s) * (a1 - t))
    s2 = np.sqrt(a2 * (a2 - q) * (a2 - r) * (a2 - t))
    x1 = ivt2[:27, 0] + evt2[:27, 0] - ivt[:27, 0] - evt[:27, 0]
    y1 = ivt2[:27, 1] + evt2[:27, 1] - ivt[:27, 1] - evt[:27, 1]
    d = 0.5 * np.sqrt(x1**2 + y1**2)
    w = c * (s1 + s2) / d

    af = np.zeros((29, 2))
    af[:27, 0] = c * d
    af[:27, 1] = 1.4 * alpha * (w**beta)

    af[27, 0] = (ivt[NP-2, 0] - ivt[NP-1, 0]) * c
    af[27, 1] = np.pi * lip_h * lip_w * cc
    af[28, 0] = af[27, 0]
    af[28, 1] = af[27, 1]

    gui['B'] = lip_h * c
    gui['A'] = lip_w * c

    tmp = np.where(af[:, 0] <= 0)[0]
    af[tmp, 0] = 0.01

    gui['area'] = af
    
    return gui

def freqevalNN(Ap, gui, valrect):
    c = 35100
    ro = 1.14e-3
    lambda_ = 5.5e-5
    eta = 1.4
    mu = 1.86e-4
    cp = 0.24
    bp = 1600
    mp = 1.4

    CONST_DAT = np.array([c, ro, lambda_, eta, mu, cp, bp, mp])

    gui['prm'][:7] = Ap
    # Voir la forme de gui la plus compatible
    gui = vlam2009NN(gui)

    # Changer la structure d'oral
    area=np.column_stack((gui['area'][:, 0], softrect(gui['area'][:, 1], valrect)))

    nbfreq = 500
    Fmax = 10000
    Fmin = Fmax / (nbfreq - 1)

    H_eq, F_form = vtn2frm_ftr_oral(area, nbfreq, Fmax, Fmin, 0, CONST_DAT)

    spec = np.abs(H_eq)

    F1 = F_form[0].item() 
    F2 = F_form[1].item() 
    F3 = F_form[2].item() 
    return F1, F2, F3, gui, spec

def arcval(pt, co, params, D, thetabounds, openi, nu, K, T, Pexp):
    if openi == 0:  # fermé-fermé
        theta = np.linspace(thetabounds[0], thetabounds[1], D)
    elif openi == -1:  # ouvert-fermé
        theta1 = np.linspace(thetabounds[0], thetabounds[1], D + 1)
        theta = theta1[1:D + 1]
    elif openi == 1:  # fermé-ouvert
        theta = np.linspace(thetabounds[0], thetabounds[1], D + 1)

    pd = pt[0, :]
    pa = pt[1, :]

    Pt = co[params, 1] + (np.cos(theta[T] / 2) ** Pexp) * pa[0] * co[params, 0] * np.cos(co[params, 2] - pa[1] - (nu / K) * theta[T]) + \
         (1 - np.cos(theta[T] / 2) ** Pexp) * pd[0] * co[params, 0] * np.cos(co[params, 2] - pd[1])

    return Pt

def pointcons2(ptc, ptv1, ptv2, co, parcons, parvoy, nu, Kvoy, K, T, delta, Pexp):
    Pt = np.zeros(7)
    Pt[parcons] = arcval(np.vstack([ptc, ptv2]), co, parcons, T, [-np.pi, 0], 0, nu, K, delta, Pexp)  # delta + 1 > delta le 09/12/2024
    Pt[parvoy] = arcval(np.vstack([ptv1, ptv2]), co, parvoy, 2 * T, [-np.pi, 0], 0, nu, Kvoy, T + delta, Pexp)
    return Pt

def pointvoy(pt, co):
    return co[:, 1] + pt[0] * co[:, 0] * np.cos(co[:, 2] - pt[1])

def consvalD(thetavoy):
    return  -4.5 * np.pi / 8

def consvalG(thetavoy):
    if thetavoy <= np.pi: # < /ga/ palatal <= /ga/ vélaire
        return [1.2, np.pi / 3] 
    else:
        return [1.1, -np.pi / 12] 

def main():

    valrect = 0.75
    Pexp = 2 
    if 0:
        decal = 0 # Fibure 4a 
    else:
        decal = 4 # Figure 4b
    coart = 0.5
    bool2D = 1
    bool3D = 0
    boolG = 0

    co = np.array([[-1.5, 0, np.pi],   
                   [-2.5, 0, -np.pi/3],
                   [3, 0, np.pi/3],
                   [-2.75, 0.5, np.pi],   
                   [3, 0, np.pi/3],
                   [2.5, 0.5, np.pi],
                   [-2, 0, np.pi/3]]) 

    art = np.array([[1, 2, 6, 0],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4]])    

    cons = np.array([[1.2, np.pi/3],   
                     [1.2, 0],
                     [1.1, 0]])
    dur = 16
    K = 10
    Kvoy = 30
    Kpause = 30
    nu = -1

    gui = initVLAMLength(np.zeros(7), 195)
    
    if 1:
        theta = [np.pi/3, np.pi/2, 2*np.pi/3, np.pi, 4*np.pi/3, 3*np.pi/2, 5*np.pi/3, 5.5*np.pi/3]
        rho = [1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.7] # retrait du /y/ 
    else:
        theta= np.linspace(np.pi/3, 5*np.pi/3, 60)
        rho = np.hstack([np.linspace(1,0.8,7), 0.80*np.ones(53), np.linspace(0.8,0.9,5)])
    
    nv = len(theta) # 8 pour 8 voyelles
    f1v = np.zeros([nv,3])
    f2v = np.zeros([nv,3])
    f3v = np.zeros([nv,3])
    f1c = np.zeros([nv,3])
    f2c = np.zeros([nv,3])
    f3c = np.zeros([nv,3])

    for k1 in range(1, nv+1): 
        cons[1, 1] = consvalD(theta[k1 - 1])
        cons[2, :] = consvalG(theta[k1 - 1])

        for k2 in range(1, 4): 
            # Voyelle
            f1v[k1 - 1, k2 - 1], f2v[k1 - 1, k2 - 1], f3v[k1 - 1, k2 - 1], gui, _ = freqevalNN(pointvoy([rho[k1 - 1], theta[k1 - 1]], co), gui, valrect)

            # Consonne            
            nmP = np.setxor1d((art[k2 - 1][art[k2 - 1] != 0]) - 1, np.arange(7))

            # Calcul en T+delta 
            Pval = pointcons2(cons[k2 - 1, :], [coart*rho[k1 - 1], theta[k1 - 1]], [rho[k1 - 1], theta[k1 - 1]], co,(art[k2 - 1][art[k2 - 1] != 0]) - 1, nmP, nu, Kvoy, K, dur, decal, Pexp)  

            f1c[k1 - 1, k2 - 1], f2c[k1 - 1, k2 - 1], f3c[k1 - 1, k2 - 1], gui, spec = freqevalNN(Pval, gui, valrect)
            
            if 0:  # affichage à t=decal
                fig = showgui(gui)
                filename = os.path.join('P:\\pythoncode\\LocusPlots\\Plotconfigessai\\', f'{(k2-1)*nv+k1:03d}.png')  # Noms : figure_001.png, figure_002.png, etc.
                plt.savefig(filename, dpi=300)
                # plt.show()
             
            
    P1f2 = np.polyfit(f2v[:, 0], f2c[:, 0], 1)
    P2f2 = np.polyfit(f2v[:, 1], f2c[:, 1], 1)
    print(P1f2) # pour obtenir les equations du locus
    print(P2f2)
    P3f2 = np.polyfit(f2v[0:int(nv/2), 2], f2c[0:int(nv/2), 2], 1)  # without /a/
    P4f2 = np.polyfit(f2v[int(nv/2):, 2], f2c[int(nv/2):, 2], 1)
    print(P3f2)
    print(P4f2)
   
    P1f3 = np.polyfit(f3v[:, 0], f3c[:, 0], 1)
    P2f3 = np.polyfit(f3v[:, 1], f3c[:, 1], 1)
    print(P1f3) # pour obtenir les equations du locus
    print(P2f3)
    P3f3 = np.polyfit(f3v[0:int(nv/2), 2], f3c[0:int(nv/2), 2], 1) # without /a/
    P4f3 = np.polyfit(f3v[int(nv/2):, 2], f3c[int(nv/2):, 2], 1)
    print(P3f3) # pour obtenir les equations du locus
    print(P4f3)

    if bool2D:
        fig=plt.figure()
        plt.plot(f2v[:, 0], f2c[:, 0], 'b.')
        plt.plot(f3v[:, 0], f3c[:, 0], 'b.')
        plt.plot(f2v[:, 1], f2c[:, 1], 'g.')
        plt.plot(f3v[:, 1], f3c[:, 1], 'g.')
        plt.plot(f2v[:, 2], f2c[:, 2], 'r.')
        plt.plot(f3v[:, 2], f3c[:, 2], 'r.')
        
        plt.plot([900, 2500], np.polyval(P1f2, [900, 2500]),'b-')
        plt.plot([900, 2500], np.polyval(P2f2, [900, 2500]),'g-')
        
        plt.plot([900, 1700], np.polyval(P3f2, [900, 1700]),'r-') # without /a/
        plt.plot([1900, 2400], np.polyval(P4f2, [1900, 2400]),'r-') 
      
        plt.plot([1900, 3800], np.polyval(P1f3, [1900, 3800]),'b-')
        plt.plot([1900, 3800], np.polyval(P2f3, [1900, 3800]),'g-')
        plt.plot([2000, 2800], np.polyval(P3f3, [2000, 2800]),'r-')  # without /a/
        plt.plot([2500, 3900], np.polyval(P4f3, [2500, 3900]),'r-')

        plt.axis([400, 4500, 400, 4500])
        plt.xlabel('F2F3v')
        plt.ylabel('F2F3c')
        plt.show()
        
    if boolG: 
        plt.plot(f2v[:, 2], f2c[:, 2], 'r.')
        plt.plot(f3v[:, 2], f3c[:, 2], 'r.')
        
        if 1:
            plt.plot([900, 1700], np.polyval(P3f2, [900, 1700]),'r-') # without /a/
            plt.plot([1900, 2400], np.polyval(P4f2, [1900, 2400]),'r-') 
          
            plt.plot([2000, 2800], np.polyval(P3f3, [2000, 2800]),'r-')  # without /a/
            plt.plot([2500, 3900], np.polyval(P4f3, [2500, 3900]),'r-')

        plt.axis([400, 4500, 400, 3500])
        plt.xlabel('F2F3v')
        plt.ylabel('F2F3c')
        plt.show()

    if bool3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Définition des limites des axes
        ax.set_xlim(2700, 700)
        ax.set_ylim(700, 2700)
        ax.set_zlim(2000, 4000)

        # Ajout des labels des axes
        ax.set_xlabel('F2v (Hz)')
        ax.set_ylabel('F2c (Hz)')
        ax.set_zlabel('F3c (Hz)')
        ax.set_title('3D Locus')

        # Tracé des points dans l'espace 3D
        ax.plot3D(f2v[:, 0], f2c[:, 0], f3c[:, 0], 'b.-')  # /b/
        ax.plot3D(f2v[:, 1], f2c[:, 1], f3c[:, 1], 'g.-')  # /d/
        ax.plot3D(f2v[:, 2], f2c[:, 2], f3c[:, 2], 'r.-')  # /g/
        ax.plot3D(f2v[:, 0], f2c[:, 0], 2000*np.ones(nv), 'b.-')  # /b/
        ax.plot3D(f2v[:, 1], f2c[:, 1], 2000*np.ones(nv), 'g.-')  # /d/
        ax.plot3D(f2v[:, 2], f2c[:, 2], 2000*np.ones(nv), 'r.-')  # /g/

        # Réglage de la vue initiale
        ax.view_init(elev=45, azim=49)

        # Affichage de la figure
        plt.show()

if __name__ == "__main__":
        main()