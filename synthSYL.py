import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import find_peaks 
from scipy.signal import lfilter
from scipy.interpolate import interp1d
from scipy.signal import medfilt
import librosa
import librosa.display
from numpy.polynomial import Polynomial
import sounddevice as sd
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# synthese de gif
from PIL import Image
import ctypes
import os

def play(signal):
        sd.play(signal, 20000)

def softrect(x, S):
    return (np.sqrt(x**2 + S) + x) / 2

def poly2rc(a):
    """
    Convertit les coefficients LPC (a) en coefficients de réflexion (RC) selon la récurrence de Levinson.

    Paramètres :
    a : ndarray
        Vecteur des coefficients LPC. Le premier élément doit être 1.
        
    Retourne :
    k : ndarray
        Vecteur des coefficients de réflexion (RC).
    """
    # Vérifier que le premier élément est bien 1
    if a[0] != 1:
        raise ValueError("Le premier coefficient de a doit être 1.")
    
    # Initialisation
    p = len(a) - 1  # Ordre des LPC
    k = np.zeros(p)  # Coefficients de réflexion
    a_current = np.copy(a[1:])  # Ignorer le premier élément (déjà égal à 1)
    
    # Boucle inversée sur les coefficients
    for i in range(p-1, -1, -1):
        # Coefficient de réflexion
        k[i] = a_current[i]
        
        # Mise à jour des coefficients a
        if i > 0:
            # Fliplr retourne le renversement des coefficients jusqu'à l'indice i
            flipped_a = np.flip(a_current[:i])
            a_current[:i] = (a_current[:i] - k[i] * flipped_a) / (1 - k[i]**2)
    
    return np.float64(k)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# % fonction de calcul d'un signal a partir d'un signal d'excitation
# % et d'un jeu de vecteurs LPC successifs (paramètres du filtre LPC)
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# % inputs : e : excitation signal
# %          LPC : matrix of LPC vectors (in lines)
# %          step_size : hop size between successive filters (in s)
# %          trans_size : size of inter-frame filter transition (in s)
# %          fs : sampling frequency
# %          LPC : matrix of LPC coefficients (can be prediction coeff., 
# %                (parcor coeff., or LSF coeff., just change the code...) 
# %
# % output : sig : speech signal
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %  Structure du filtre et notations
# %  Les branches croisees sont orientees de droite a gauche
# %
# %
# %  e(n)=e_f(p+1)           e_f(3)          e_f(2) 
# %     -->(+)------------ ... --->(+)-------------(+)----------------> s(n) = e_f(1)
# %         - \  /                  - \  /          - \  /         |
# %            \/ kp                   \/  k2          \/ k1       |
# %            /\                      /\              /\          |
# %           /  \                    /  \            /  \         |
# %     <--(+)<---- z-1 -- ... ----(+)<---- z-1 ---(+)<---- z-1 <--
# %  e_b(p+1)                e_b(3)            e_b(2)              e_b(1) = s(n)
# %  (inutile)
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
def f_lpc_exc2sig(e, step_size, trans_size, fs, LPC):
    N_step = int(step_size * fs)
    N_trans = int(trans_size * fs)

    L = len(e)
    M = L // N_step
    if M != LPC.shape[0]:
        print('\nProblem with the size of the data\n')
        return 0

    p = LPC.shape[1] - 1
    sig = np.zeros(L)
    e_f = np.zeros(p + 1)
    e_b = np.zeros(p + 1)

    a = LPC[0, :]
    k_P = poly2rc(a)
    
    for m in range(M):
        a1 = LPC[m, :]
        k_C = poly2rc(a1)

        dk = (k_C - k_P) / N_trans
        
        for ind in range(N_step):
            if ind < N_trans:
                k_P += dk
            
            e_f[p] = e[ind + m * N_step]
            
            for ind_p in range(p):
                    e_f[p - ind_p - 1] = e_f[p - ind_p] - k_P[p - ind_p - 1] * e_b[p - ind_p - 1]
                    e_b[p - ind_p] = e_b[p - ind_p - 1] + k_P[p - ind_p - 1] * e_f[p - ind_p - 1]
            
            e_b[0] = e_f[0]
            sig[ind + m * N_step] = e_f[0]  
    
    return sig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# %	Function de génération de signal source v.2.0
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# %	date : 11 mai 2009
# %   derniere mise a jour : idem
# %	auteur : Laurent Girin - Gipsa-lab DPC
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# % Inputs : L : length of generated signal
# %          fs: sampling frequency
# %          F0: fundamental frequency trajectory (in Hz)
# %
# % Output : s : source signal
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def gen_src_3(L, fs, F0, flag_stretch):
    M = len(F0)
    if M != L:
        F0 = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, M), F0)

    if flag_stretch == 0:
        F0_max = np.max(F0)
        T = int(fs / F0_max / 0.8)
        Tp = int(T / 3)
        Tn = int(T / 4)
        t1 = np.arange(1, Tp + 1)
        t2 = np.arange(Tp + 1, Tp + Tn + 1)
        pulse = np.concatenate([(3 * (t1 / Tp) ** 2 - 2 * (t1 / Tp) ** 3), (1 - ((t2 - Tp) / Tn) ** 2)])
        train = np.zeros(L)
        ind = 0
        while True:
            train[ind] = 1
            ind += int(fs / F0[ind])
            if ind >= L:
                break
        s = np.convolve(train, pulse, mode='full')

    else:
        s = np.zeros(L)
        ind = 0
        while True:
            T = int(fs / F0[ind])
            Tp = int(T / 3)
            Tn = int(T / 6)
            t1 = np.arange(1, Tp + 1)
            t2 = np.arange(Tp + 1, Tp + Tn + 1)
            pulse = np.concatenate([(3 * (t1 / Tp) ** 2 - 2 * (t1 / Tp) ** 3), (1 - ((t2 - Tp) / Tn) ** 2)])
            s[ind:ind + Tp + Tn] = pulse
            ind += int(fs / F0[ind])
            if ind >= L:
                break

    return s[:L]

def levinson_durbin(r, p):
    """
    Implémentation de l'algorithme de Levinson-Durbin avec la même sortie que MATLAB.
    Le premier élément du vecteur de sortie est 1.
    
    Paramètres :
    r : ndarray
        Coefficients d'autocorrélation.
    p : int
        Ordre du modèle LPC.
    
    Retourne :
    a : ndarray
        Coefficients du modèle LPC, avec le premier élément égal à 1.
    e : float
        Erreur quadratique résiduelle.
    """
    # Initialisation
    a = np.float64(np.zeros(p + 1))  # Inclut le coefficient d'ordre zéro qui sera fixé à 1
    e = r[0]  # Erreur initiale

    if e == 0:  # Si l'autocorrélation initiale est nulle
        a[0] = 1  # Premier coefficient doit être 1
        return a, e

    a[0] = 1  # Le premier coefficient du modèle LPC doit être 1 (comme MATLAB)
    
    for i in range(1, p + 1):
        # Calcul du coefficient de réflexion (reflection coefficient)
        if i == 1:
            k = r[1] / e
        else:
            k = (r[i] - np.dot(a[1:i], r[i-1:0:-1])) / e

        # Mise à jour des coefficients du modèle LPC
        a_new = a[1:i] - k * np.flip(a[1:i])  # Nouvelle valeur pour a[1:i]
        a[i] = k  # Dernier coefficient mis à jour avec le coefficient de réflexion k
        a[1:i] = a_new  # Mise à jour de a[1:i]
        
        # Mise à jour de l'erreur quadratique résiduelle
        e *= (1 - k**2)
        if e < 0:  # Pour éviter des erreurs mathématiques dues à des valeurs négatives
            e = 0
            
    a[1:] = -a[1:]
    return np.float64(a), e

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# % Calcul de filtre LPC A(z) a partir des echantillons 
# % de la reponse en frequence
# %
# % Date : 13 mars 2009
# %
# % Auteur : L. Girin - GIPSA-lab
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# % entrée : H : vecteur des echantillons de la reponse en frequence
# % (seulement pour N frequences positives, de 0 à (N-1)/N*fe/2)
# %
# % Sortie : a : vecteur de coefficients LPC
# %          g : gain associé
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def Hfreq2lpc(H, p):
    """
    Convertit un spectre de fréquence en un modèle LPC.

    Paramètres :
    H_eq : ndarray
        Le spectre d'entrée (peut être en échelle linéaire).
    p : int
        Ordre du modèle LPC.

    Retourne :
    g : float
        Le gain du modèle LPC.
    a : ndarray
        Les coefficients du modèle LPC.
    """

    # Assurez-vous que H est un vecteur colonne
    H = np.asarray(H).flatten()  # Equivalent de H(:) en Matlab
    N = len(H)

    # Complétion avec les "fréquences négatives"
    Hr = np.concatenate([H, [H[N-1]], np.flipud(H[1:N-1])])

    # Coefficients d'autocorrélation = IFFT(DSP)
    R = np.float64(np.real(np.fft.ifft(np.abs(Hr) ** 2)))

    # p+1 coefficients pour un modèle d'ordre p
    R = R[:p+1]
 
    # Utilisation de la fonction Levinson-Durbin pour calculer les coefficients LPC
    a, e = levinson_durbin(R, p)

    # Gain LPC
    g = np.sqrt(e)
    
    return g, np.float64(a)

def synthsimpleWORDfen(matspec, word, cf0, cexp, cdec, dur):
    fs = 20000

    lg, _ = matspec.shape
    M = lg
    p = 30
    A = np.zeros((p + 1, 1))

    T_s = 2 * 5e-3
    L_t = M * T_s
    N_t = int(np.fix(L_t * fs))
    A = []

    for k in range(lg):
        G, Ai = Hfreq2lpc(matspec[k, :], p)
        A.append(Ai.flatten())

    F0 = interp1d([1, int(np.fix(N_t / 3)), int(2 * np.fix(N_t / 3)), N_t],
              np.array([100, 130, 110, 90]),
              kind='cubic')(np.arange(1, N_t + 1))

    np.random.seed(0)
    F0 += np.random.randn(*F0.shape) * np.mean(F0) / 100 * 0.5  # jitter
    e = gen_src_3(N_t, fs, F0, 0)
    sig = f_lpc_exc2sig(e, T_s, 1e-3, fs, np.array(A))

    lg = len(sig)
    lg2 = round(lg / 4)  
    han = np.hanning(lg2)
    handeb = han[:round(lg2 / 2)]
    hanfin = han[round(lg2 / 2):]
    fen = np.concatenate([handeb, np.ones(lg - lg2), hanfin ** 2])
    fen2 = synthfen(word, cexp, cdec, int(dur * fs * T_s))  
    sig = lfilter([1, -0.9375], 1, fen * fen2 * sig)

    sig = sig / (1.01 * np.max(np.abs(sig)))

    play(sig)  # Uncomment this line if you want to play the sound
    return sig

def voysynth(spec):
    fs = 20000
 
    M = 100
    p = 30
    A = np.zeros((p + 1, 1))
    T_s = 5e-3
    L_t = M * T_s
    N_t = int(np.fix(L_t * fs))
    G, A = Hfreq2lpc(spec, p)
    A = A.reshape(-1,1)
    
    # Ajout d'un quatrième point
    F0 = interp1d([1, int(np.fix(N_t / 3)), int(2 * np.fix(N_t / 3)), N_t],
              np.array([100, 130, 110, 90]),
              kind='cubic')(np.arange(1, N_t + 1))
    
    np.random.seed(0)
    F0 += np.random.randn(*F0.shape) * np.mean(F0) / 100 * 0.5
    
    e = gen_src_3(N_t, fs, F0, 0)
    sig = f_lpc_exc2sig(e, T_s, 1e-3, fs, (A * np.ones((1, M))).T)
    
    han = np.hanning(len(sig))
    sig = lfilter([1, -0.9375], 1, (han * sig).T).T
    sig = sig / (1.01 * np.max(np.abs(sig)))
    
    return sig

def synthsimpleSYL(matspec, syl, cf0, cexp, cdec):
    fs = 20000
    lg, _ = matspec.shape
    
    M = lg
    p = 30
    A = np.zeros((p + 1, 1))

    T_s = 2 * 5e-3
    L_t = M * T_s
    N_t = int(np.fix(L_t * fs))

    A = []
    for k in range (lg):
        G, Ai = Hfreq2lpc(matspec[k, :], p)
        A.append(Ai.flatten())

    F0 = interp1d([1, int(np.fix(N_t / 3)), int(2 * np.fix(N_t / 3)), N_t],
              np.array([100, 130, 110, 90]),
              kind='cubic')(np.arange(1, N_t + 1))

    np.random.seed(0)
    F0 = F0 + np.random.randn(*F0.shape) * np.mean(F0) / 100 * 0.5
    e = gen_src_3(N_t, fs, F0, 0)    
    sig = f_lpc_exc2sig(e, T_s, 1e-3, fs, np.array(A))

    lg = len(sig)
    lg2 = round(lg / 4)
    han = np.hanning(lg2)
    handeb = han[:round(lg2 / 2)]
    hanfin = han[round(lg2 / 2):]
    fen = np.concatenate((handeb, np.ones(lg - lg2), hanfin ** 2))
    longue = 0

    if syl == 'CV':
        if longue:
            fen3 = np.hanning(round(lg / 2))
            fen2 = np.concatenate((np.zeros(round(lg / 4)), fen3[:round(lg / 4)] ** cexp, np.ones(round(lg / 2))))
        else:
            fen3 = np.hanning(round(lg / 1.5))
            fen2 = np.concatenate((np.zeros(round(lg / 3)), fen3[:round(lg / 3)] ** cexp, np.ones(round(lg / 3))))
    elif syl == 'CVV':
        fen3 = np.hanning(round(lg / 2))
        fen2 = np.concatenate((np.zeros(round(lg / 4)), fen3[:round(lg / 4)] ** cexp, np.ones(round(lg / 2))))
    elif syl == 'VCVV':
        fen3 = np.hanning(round(lg / 2.5))
        fen2 = np.concatenate((np.ones(round(lg / 5)), fen3[round(lg / 5):] ** cdec, fen3[:round(lg / 5)] ** cexp, np.ones(round(lg / 2.5))))
    elif syl == 'VC':
        fen3 = np.hanning(round(lg / 2))
        fen2 = np.concatenate((np.ones(round(lg / 2)), fen3[round(lg / 4):] ** cdec, fen3[:round(lg / 4)] ** cexp))
    elif syl == 'VVC':
        fen3 = np.hanning(round(lg / 2))
        fen2 = np.concatenate((np.ones(round(lg / 2)), fen3[round(lg / 4):] ** cdec, fen3[:round(lg / 4)] ** cexp))
    elif syl == 'VCV':
        fen3 = np.hanning(round(lg / 2))
        fen2 = np.concatenate((np.ones(round(lg / 4)), fen3[round(lg / 4):] ** cdec, fen3[:round(lg / 4)] ** cexp, np.ones(round(lg / 4))))
    elif syl == 'CVC':
        if longue:
            fen3 = np.hanning(round(lg / 2.5))
            fen2 = np.concatenate((np.zeros(lg // 5), fen3[:round(lg / 5)] ** cexp, np.ones(round(lg / 5)), fen3[round(lg / 5):] ** cdec, fen3[:round(lg / 5)] ** cexp))
        else:
            fen3 = np.hanning(round(lg / 2))
            fen2 = np.concatenate((np.zeros(round(lg / 4)), fen3[:round(lg / 4)] ** cexp, fen3[round(lg / 4):] ** cdec, fen3[:round(lg / 4)] ** cexp))
    elif syl == 'VCVC':
        fen3 = np.hanning(round(lg / 3))
        fen2 = np.concatenate((np.ones(round(lg / 6)), fen3[round(lg / 6):] ** cdec, fen3[:round(lg / 6)] ** cexp, np.ones(round(lg / 6)), fen3[round(lg / 6):] ** cdec, fen3[:round(lg / 6)] ** cexp))
    elif syl == 'CVVC':
        fen3 = np.hanning(round(lg / 3))
        fen2 = np.concatenate((np.zeros(round(lg / 6)), fen3[:round(lg / 6)] ** cexp, np.ones(round(lg / 3)), fen3[round(lg / 6):] ** cdec, fen3[:round(lg / 6)] ** cexp))
    elif syl == 'VCVVC':
        fen3 = np.hanning(round(lg / 3.5))
        fen2 = np.concatenate((np.ones(round(lg / 7)), fen3[round(lg / 7):] ** cdec, fen3[:round(lg / 7)] ** cexp, np.ones(round(lg / 3.5)), fen3[round(lg / 7):] ** cdec, fen3[:round(lg / 7)] ** cexp))
    elif syl == 'CCV':
        fen3 = np.hanning(round(lg / 2))
        fen2 = np.concatenate((np.zeros(round(lg / 4)), (fen3[:round(lg / 4)] ** cexp) * fen3[round(lg / 4):] ** cdec, fen3[:round(lg / 4)] ** cexp, np.ones(round(lg / 4))))
    elif syl == 'CCVV':
        fen3 = np.hanning(round(lg / 2.5))
        fen2 = np.concatenate((np.zeros(round(lg / 5)), (fen3[:round(lg / 5)] ** cexp) * fen3[round(lg / 5):] ** cdec, fen3[:round(lg / 5)] ** cexp, np.ones(round(lg / 5)), np.ones(round(lg / 5))))
    elif syl == 'VCCVV':
        fen3 = np.hanning(round(lg / 3))
        fen2 = np.concatenate((np.ones(round(lg / 6)), fen3[round(lg / 6):] ** cdec, (fen3[:round(lg / 6)] ** cexp) * fen3[round(lg / 6):] ** cdec, fen3[:round(lg / 6)] ** cexp, np.ones(round(lg / 6)), np.ones(round(lg / 6))))
    elif syl == 'VCC':
        fen3 = np.hanning(round(lg / 2))
        fen2 = np.concatenate((np.ones(round(lg / 4)), fen3[round(lg / 4):] ** cdec, (fen3[:round(lg / 4)] ** cexp) * fen3[round(lg / 4):] ** cdec, fen3[:round(lg / 4)] ** cexp))
    elif syl == 'VVCC':
        fen3 = np.hanning(round(lg / 2.5))
        fen2 = np.concatenate((np.ones(round(lg / 5)), np.ones(round(lg / 5)), fen3[round(lg / 5):] ** cdec, (fen3[:round(lg / 5)] ** cexp) * fen3[round(lg / 5):] ** cdec, fen3[:round(lg / 5)] ** cexp))
    elif syl == 'VCCV':
        fen3 = np.hanning(round(lg / 2.5))
        fen2 = np.concatenate((np.ones(round(lg / 5)), fen3[round(lg / 5):] ** cdec, (fen3[:round(lg / 5)] ** cexp) * fen3[round(lg / 5):] ** cdec, fen3[:round(lg / 5)] ** cexp, np.ones(round(lg / 5))))

    sig = lfilter([1, -0.9375], 1, fen * fen2 * sig)
    sig = sig / (1.01 * np.max(np.abs(sig)))
    play(sig) 
    return sig


def synthpause(gui, Pval, valrect):
    f1, f2, f3 = [], [], []

    for k in range(len(Pval)):
        f1_k, f2_k, f3_k, _, _ = freqevalNN(Pval[k, :], gui, valrect)
        f1.append(f1_k)
        f2.append(f2_k)
        f3.append(f3_k)

    f = np.array([f1, f2, f3]).T
    return f

def synthfen1(word1, cexp, cdec, durfen):
    word = ''.join(word1)
    nbfen = len(word) - 1
    fen3 = np.hanning(2 * durfen)
    fen2 = np.array([])

    for k in range(nbfen):
        syl = word[k:k + 2]
        if syl == 'OC':
            fen2 = np.concatenate((fen2, np.zeros(durfen)))
        elif syl == 'OV':
            fen2 = np.concatenate((fen2, np.ones(durfen)))
        elif syl == 'CF':
            fen2 = np.concatenate((fen2, fen3[:durfen] ** cexp))
        elif syl == 'VF':
            # fen2 = np.concatenate([fen2, np.ones(durfen)])
            pass  # fen2.extend([1] * durfen)  # Uncomment if needed  
        elif syl == 'VV':
            fen2 = np.concatenate((fen2, np.ones(durfen)))
        elif syl in ['CV', 'cV']:
            fen2 = np.concatenate((fen2, fen3[:durfen] ** cexp, np.ones(durfen)))
        elif syl in ['VC', 'Vc']:
            fen2 = np.concatenate((fen2, fen3[durfen:] ** cdec))
        elif syl == 'CC':
            fen2 = np.concatenate((fen2, 0.5 * fen3[:durfen] ** cexp, 0.5 * fen3[durfen:] ** (3 * cdec)))
        elif syl in ['Cc', 'cC']:
            fen2 = np.concatenate((fen2, fen3[:durfen] ** cexp) * (fen3[durfen:] ** (3 * cdec)))
    return fen2

def synthfen(word1, cexp, cdec, durfen):
    word = ''.join(word1)
    nbfen = len(word) - 1
    fen3 = np.hanning(2 * durfen)
    fen2 = np.array([])

    for k in range(nbfen):
        syl = word[k:k + 2]
        if syl == 'OC':
            fen2 = np.concatenate([fen2, np.zeros(durfen)])
        elif syl == 'OV':
            fen2 = np.concatenate([fen2, np.ones(durfen)])
        elif syl == 'CF':
            fen2 = np.concatenate([fen2, fen3[:durfen] ** cexp])
        elif syl == 'VF':
            # À implémenter si nécessaire
            pass
        elif syl == 'VV':
            fen2 = np.concatenate([fen2, np.ones(durfen)])
        elif syl in ['CV', 'cV']:
            fen2 = np.concatenate([fen2, fen3[:durfen] ** cexp, np.ones(durfen)])
        elif syl in ['VC', 'Vc']:
            fen2 = np.concatenate([fen2, fen3[durfen:] ** cdec])
        elif syl == 'CC':
            fen2 = np.concatenate([
                fen2, 
                0.5 * fen3[:durfen] ** cexp, 
                0.5 * fen3[durfen:] ** (3 * cdec)
            ])
        elif syl in ['Cc', 'cC']:
            product = (fen3[:durfen] ** cexp) * (fen3[durfen:] ** (3 * cdec))
            fen2 = np.concatenate([fen2, product])
    
    return fen2

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
   
def synthsyl(gui, Pval, syl, cf0, valrect):
    f1, f2, f3 = [], [], [],
    matspec = []
    
    for k in range(len(Pval)):
        f1_k, f2_k, f3_k, gui, spec = freqevalNN(Pval[k, :], gui, valrect)
        f1.append(f1_k)
        f2.append(f2_k)
        f3.append(f3_k)
        matspec.append(spec)

    f = np.array([f1, f2, f3])
    sig = synthsimpleSYL(np.array(matspec), syl['typ'].upper(), cf0, 1, 3)
    return sig, f

def synthwordfen(gui, Pval, word, cf0, valrect, dur):
    f1, f2, f3 = [], [], [],
    matspec = []
    
    for k in range(len(Pval)):
        f1_k, f2_k, f3_k, gui, spec = freqevalNN(Pval[k, :], gui, valrect)
        f1.append(f1_k)
        f2.append(f2_k)
        f3.append(f3_k)
        matspec.append(spec)
    
    f = np.array([f1, f2, f3])
    sig = synthsimpleWORDfen(np.array(matspec), word, cf0, 1, 2, dur)
    return sig, f


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
    gui = vlam2009NN(gui)

    area=np.column_stack((gui['area'][:, 0], softrect(gui['area'][:, 1], valrect)))

    nbfreq = 500
    Fmax = 10000
    Fmin = Fmax / (nbfreq - 1)

    H_eq, F_form = vtn2frm_ftr_oral(area, nbfreq, Fmax, Fmin, 0, CONST_DAT)

    spec = np.abs(H_eq)

    F1 = F_form[0] 
    F2 = F_form[1] 
    F3 = F_form[2] 
    return F1, F2, F3, gui, spec

def pointvoy(pt, co):
    return co[:, 1] + pt[0] * co[:, 0] * np.cos(co[:, 2] - pt[1])

def pointplot(pt, D):
    Tt = np.zeros((D, 2), dtype=complex)
    for k in range(D):
        Tt[k, 0] = pt[0] * np.cos(pt[1]) + 1j * pt[0] * np.sin(pt[1])
    Tt[:, 1] = Tt[:, 0]
    return Tt

def point(pt, co, D):
    Pt = np.zeros((D,co.shape[0]))
    for k in range(D):
        Pt[k,:] = co[:, 1] + pt[0] * co[:, 0] * np.cos(co[:, 2] - pt[1])
    return Pt

def arcplot(pt, D, thetabounds, opint, nu, K, Pexp):
    if opint == 0:  # fermé-fermé
        theta = np.linspace(thetabounds[0], thetabounds[1], D)
    elif opint == -1:  # ouvert-fermé
        theta1 = np.linspace(thetabounds[0], thetabounds[1], D + 1)
        theta = theta1[1:D + 1]
    elif opint == 1:  # fermé-ouvert
        theta = np.linspace(thetabounds[0], thetabounds[1], D + 1)

    pd = pt[0, :]
    pa = pt[1, :]
    Tt = np.zeros(D, dtype=complex)

    for k in range(D):
        rho = np.cos(theta[k] / 2) ** Pexp
        
        Tt[k] = (rho * pa[0] * np.cos(pa[1] + (nu / K) * theta[k]) +
                     (1 - rho) * pd[0] * np.cos(pd[1])) + \
                    1j * (rho * pa[0] * np.sin(pa[1] + (nu / K) * theta[k]) +
                          (1 - rho) * pd[0] * np.sin(pd[1]))

    return Tt

def arc(pt, co, params, D, thetabounds, opint, nu, K, Pexp):
    if opint == 0:  # fermé-fermée
        theta = np.linspace(thetabounds[0], thetabounds[1], D)
    elif opint == -1:  # ouvert-fermée
        theta1 = np.linspace(thetabounds[0], thetabounds[1], D + 1)
        theta = theta1[1:D + 1]
    elif opint == 1:  # fermé-ouvert
        theta = np.linspace(thetabounds[0], thetabounds[1], D + 1)

    pd = pt[0, :]
    pa = pt[1, :]

    Pt = np.zeros((D,len(params)))
    for k in range(D):
        rho = np.cos(theta[k] / 2) ** Pexp
        Pt[k,:] = co[params, 1] + rho * pa[0] * co[params, 0] * np.cos(co[params, 2] - pa[1] - (nu / K) * theta[k]) + \
            (1 - rho) * pd[0] * co[params, 0] * np.cos(co[params, 2] - pd[1])

    return Pt

def boucle(syl, co, nu, K, Kvoy, T, Pexp):
    nbP, _ = co.shape
    if syl['typ'].upper() == 'CV':
        Pval = np.vstack((makeloop(2, syl['P'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), 
                          point(syl['pt'][2, :], co, T)))
    elif syl['typ'].upper() == 'CVV':
        Pval = np.vstack((makeloop(2, syl['P'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), 
                          arc(syl['pt'][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCVV':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), 
                          makeloop(2, syl['P'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), 
                          arc(syl['pt'][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VC':
        Pval = np.vstack((point(syl['pt'][0, :], co, 2 * T), 
                          makeloop(2, syl['P'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VVC':
        Pval = np.vstack((arc(syl['pt'][0:2, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp), 
                          makeloop(2, syl['P'], syl['pt'][1:4, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCV':
        Pval = np.vstack((point(syl['pt'][0], co, T), 
                          makeloop(2, syl['P'], syl['pt'][0:3], nbP, co, T, nu, K, Kvoy, Pexp), 
                          point(syl['pt'][2], co, T)))
    elif syl['typ'].upper() == 'CVC':
        Pval = np.vstack((makeloop(2, syl['P1'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), 
                          makeloop(2, syl['P2'], syl['pt'][2:5, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCVC':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), 
                          makeloop(2, syl['P1'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), 
                          point(syl['pt'][2, :], co, T), 
                          makeloop(2, syl['P2'], syl['pt'][2:5, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'CVVC':
        Pval = np.vstack((makeloop(2, syl['P1'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), 
                          arc(syl['pt'][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp), 
                          makeloop(2, syl['P2'], syl['pt'][3:6, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCVVC':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), 
                          makeloop(2, syl['P1'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), 
                          arc(syl['pt'][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp), 
                          makeloop(2, syl['P2'], syl['pt'][3:6, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'CCV':
        Pval = np.vstack((makeloop(3, syl['P'], syl['pt'], nbP, co, T, nu, K, Kvoy, Pexp), 
                          point(syl['pt'][3, :], co, T)))
    elif syl['typ'].upper() == 'CCVV':
        Pval = np.vstack((makeloop(3, syl['P'], syl['pt'], nbP, co, T, nu, K, Kvoy, Pexp), 
                          arc(syl['pt'][3:5, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCCVV':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), 
                          makeloop(3, syl['P'], syl['pt'], nbP, co, T, nu, K, Kvoy, Pexp), 
                          arc(syl['pt'][3:5, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCC':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), 
                          makeloop(3, syl['P'], syl['pt'], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VVCC':
        Pval = np.vstack((arc(syl['pt'][0:2, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp), 
                          makeloop(3, syl['P'], syl['pt'][1:5, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCCV':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), 
                          makeloop(3, syl['P'], syl['pt'], nbP, co, T, nu, K, Kvoy, Pexp), 
                          point(syl['pt'][3, :], co, T)))
    
    return np.array(Pval)

def boucleword(syl, co, nu, K, Kvoy, T, Pexp):
    nbP, _ = co.shape

    if syl['typ'].upper() == 'CV':
        Pval = np.vstack((makeloop(2, syl['P'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), point(syl['pt'][2, :], co, T)))
    elif syl['typ'].upper() == 'CVV':
        Pval = np.vstack((makeloop(2, syl['P'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), arc(syl['pt'][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCVV':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), makeloop(2, syl['P'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), arc(syl['pt'][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VC':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), makeloop(2, syl['P'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VVC':
        Pval = np.vstack((arc(syl['pt'][0:2, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makeloop(2, syl['P'], syl['pt'][1:4, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCV':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), makeloop(2, syl['P'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), point(syl['pt'][2, :], co, T)))
    elif syl['typ'].upper() == 'CVC':
        Pval = np.vstack((makeloop(2, syl['P1'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), point(syl['pt'][2, :], co, T), makeloop(2, syl['P2'], syl['pt'][2:5, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCVC':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), makeloop(2, syl['P1'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), point(syl['pt'][2, :], co, T), makeloop(2, syl['P2'], syl['pt'][2:5, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'CVVC':
        Pval = np.vstack((makeloop(2, syl['P1'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), arc(syl['pt'][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makeloop(2, syl['P2'], syl['pt'][3:6, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCVVC':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), makeloop(2, syl['P1'], syl['pt'][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp), arc(syl['pt'][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makeloop(2, syl['P2'], syl['pt'][3:6, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'CCV':
        Pval = np.vstack((makeloop(3, syl['P'], syl['pt'], nbP, co, T, nu, K, Kvoy, Pexp), point(syl['pt'][3, :], co, T)))
    elif syl['typ'].upper() == 'CCVV':
        Pval = np.vstack((makeloop(3, syl['P'], syl['pt'], nbP, co, T, nu, K, Kvoy, Pexp), arc(syl['pt'][3:5, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCCVV':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), makeloop(3, syl['P'], syl['pt'], nbP, co, T, nu, K, Kvoy, Pexp), arc(syl['pt'][3:5, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCC':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), makeloop(3, syl['P'], syl['pt'], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VVCC':
        Pval = np.vstack((arc(syl['pt'][0:2, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makeloop(3, syl['P'], syl['pt'][1:5, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCCV':
        Pval = np.vstack((point(syl['pt'][0, :], co, T), makeloop(3, syl['P'], syl['pt'], nbP, co, T, nu, K, Kvoy, Pexp), point(syl['pt'][3, :], co, T)))

    return np.array(Pval)

def makeloop(n, sylP1, sylpt, nbP, co, T, nu, K, Kvoy, Pexp):
    sylP=sylP1-1 
    nmP = np.setxor1d(sylP, np.arange(nbP)) 
    Pval = np.zeros((n * T, nbP))
    if n == 2:
        Pval[:, sylP] = np.vstack((arc(sylpt[[1, 0], :], co, sylP, T, np.array([0, np.pi]), 0, nu, K, Pexp), 
                                   arc(sylpt[[1, 2], :], co, sylP, T, np.array([-np.pi, 0]), -1, nu, K, Pexp)))
        if nmP.size > 0:
            Pval[:, nmP] = arc(sylpt[[0, 2], :], co, nmP, n * T, np.array([-np.pi, 0]), 0, nu, Kvoy, Pexp)
    
    if n == 3:
        Pval[:, sylP] = np.vstack((arc(sylpt[[1, 0], :], co, sylP, T, np.array([0, np.pi]), 0, nu, K, Pexp), 
                                   arc(sylpt[[1, 2], :], co, sylP, T, np.array([-np.pi, 0]), -1, nu, K, Pexp), 
                                   arc(sylpt[[2, 3], :], co, sylP, T, np.array([-np.pi, 0]), -1, nu, K, Pexp)))
        if nmP.size > 0:
            Pval[:, nmP] = arc(sylpt[[0, 3], :], co, nmP, n * T, np.array([-np.pi, 0]), 0, nu, Kvoy, Pexp)
    return Pval


def arcplotV(pt, D, thetabounds, open, nu, K, Pexp):
    if open == 0:  # fermé-fermé
        theta = np.linspace(thetabounds[0], thetabounds[1], D)
    elif open == -1:  # ouvert-fermé
        theta1 = np.linspace(thetabounds[0], thetabounds[1], D + 1)
        theta = theta1[1:D + 1]
    elif open == 1:  # fermé-ouvert
        theta = np.linspace(thetabounds[0], thetabounds[1], D + 1)

    pd = pt[0, :]
    pa = pt[1, :]
    Tt = np.zeros((D, 2), dtype=complex)

    for k in range(D):
        rho = np.cos(theta[k] / 2) ** Pexp
        
        Tt[k, 0] = (rho * pa[0] * np.cos(pa[1] + (nu / K) * theta[k]) +
                     (1 - rho) * pd[0] * np.cos(pd[1])) + \
                    1j * (rho * pa[0] * np.sin(pa[1] + (nu / K) * theta[k]) +
                          (1 - rho) * pd[0] * np.sin(pd[1]))

    Tt[:, 1] = Tt[:, 0]
    return Tt

def boucleplot(syl, nu, K, Kvoy, T, Pexp):
    if syl['typ'].upper() == 'CV':
        Tval = np.vstack((makelooplot(2, syl['pt'][0:3, :], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][2, :], 2 * T)))
    elif syl['typ'].upper() == 'CVV':
        Tval = np.vstack((makelooplot(2, syl['pt'][0:3, :], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][2:4, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCVV':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(2, syl['pt'][0:3, :], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][2:4, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VC':
        Tval = np.vstack((pointplot(syl['pt'][0, :], 2 * T), makelooplot(2, syl['pt'][0:3, :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VVC':
        Tval = np.vstack((arcplotV(syl['pt'][0:2, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makelooplot(2, syl['pt'][1:4, :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCV':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(2, syl['pt'][0:3, :], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][2, :], T)))
    elif syl['typ'].upper() == 'CVC':
        Tval = np.vstack((makelooplot(2, syl['pt'][0:3, :], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][2, :], T), makelooplot(2, syl['pt'][2:5, :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCVC':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(2, syl['pt'][0:3, :], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][2, :], T), makelooplot(2, syl['pt'][2:5, :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'CVVC':
        Tval = np.vstack((makelooplot(2, syl['pt'][0:3, :], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][2:4, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makelooplot(2, syl['pt'][3:6, :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCVVC':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(2, syl['pt'][0:3, :], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][2:4, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makelooplot(2, syl['pt'][3:6, :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'CCV':
        Tval = np.vstack((makelooplot(3, syl['pt'], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][3, :], T)))
    elif syl['typ'].upper() == 'CCVV':
        Tval = np.vstack((makelooplot(3, syl['pt'], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][3:5, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCCVV':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(3, syl['pt'], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][3:5, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCC':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(3, syl['pt'], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VVCC':
        Tval = np.vstack((arcplotV(syl['pt'][0:2, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makelooplot(3, syl['pt'][1:5, :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCCV':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(3, syl['pt'], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][3, :], T)))
    return Tval

def bouclewordplot(syl, nu, K, Kvoy, T, Pexp):
    if syl['typ'].upper() == 'CV':
        Tval = np.vstack((makelooplot(2, syl['pt'][[0, 1, 2], :], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][2, :], T)))
    elif syl['typ'].upper() == 'CVV':
        Tval = np.vstack((makelooplot(2, syl['pt'][[0, 1, 2], :], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][[2, 3], :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCVV':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(2, syl['pt'][[0, 1, 2], :], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][[2, 3], :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VC':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(2, syl['pt'][[0, 1, 2], :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VVC':
        Tval = np.vstack((arcplotV(syl['pt'][[0, 1], :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makelooplot(2, syl['pt'][[1, 2, 3], :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCV':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(2, syl['pt'][[0, 1, 2], :], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][2, :], T)))
    elif syl['typ'].upper() == 'CVC':
        Tval = np.vstack((makelooplot(2, syl['pt'][[0, 1, 2], :], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][2, :], T), makelooplot(2, syl['pt'][[2, 3, 4], :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCVC':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(2, syl['pt'][[0, 1, 2], :], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][2, :], T), makelooplot(2, syl['pt'][[2, 3, 4], :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'CVVC':
        Tval = np.vstack((makelooplot(2, syl['pt'][[0, 1, 2], :], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][[2, 3], :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makelooplot(2, syl['pt'][[3, 4, 5], :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCVVC':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(2, syl['pt'][[0, 1, 2], :], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][[2, 3], :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makelooplot(2, syl['pt'][[3, 4, 5], :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'CCV':
        Tval = np.vstack((makelooplot(3, syl['pt'], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][3, :], T)))
    elif syl['typ'].upper() == 'CCVV':
        Tval = np.vstack((makelooplot(3, syl['pt'], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][[3, 4], :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCCVV':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(3, syl['pt'], T, nu, K, Kvoy, Pexp), arcplotV(syl['pt'][[3, 4], :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif syl['typ'].upper() == 'VCC':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(3, syl['pt'], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VVCC':
        Tval = np.vstack((arcplotV(syl['pt'][[0, 1], :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp), makelooplot(3, syl['pt'][[1, 2, 3, 4], :], T, nu, K, Kvoy, Pexp)))
    elif syl['typ'].upper() == 'VCCV':
        Tval = np.vstack((pointplot(syl['pt'][0, :], T), makelooplot(3, syl['pt'], T, nu, K, Kvoy, Pexp), pointplot(syl['pt'][3, :], T)))
    return Tval

def makelooplot(n, sylpt, T, nu, K, Kvoy, Pexp):
    Tval = np.zeros((n * T, 2), dtype=complex)
    if n == 2:
        Tval[:, 0] = np.concat((arcplot(sylpt[[1, 0], :], T, [0, np.pi], 0, nu, K, Pexp), 
                                 arcplot(sylpt[[1, 2], :], T, [-np.pi, 0], -1, nu, K, Pexp)))
        Tval[:, 1] = arcplot(sylpt[[0, 2], :], n * T, [-np.pi, 0], 0, nu, Kvoy, Pexp)
    if n == 3:
        Tval[:, 0] = np.concat((arcplot(sylpt[[1, 0], :], T, [0, np.pi], 0, nu, K, Pexp), 
                                 arcplot(sylpt[[1, 2], :], T, [-np.pi, 0], -1, nu, K, Pexp), 
                                 arcplot(sylpt[[2, 3], :], T, [-np.pi, 0], -1, nu, K, Pexp)))
        Tval[:, 1] = arcplot(sylpt[[0, 3], :], n * T, [-np.pi, 0], 0, nu, Kvoy, Pexp)
    return Tval

def playVLAMvid(output_video,fps, gui, Pval, valrect):
    # Paramètres pour l'objet VideoWriter (OpenCV)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec pour AVI
    video = None  # VideoWriter sera initialisé après la première frame

    for k in range(len(Pval)):
            # if k % 2 == 0:   # and boolvid:
            _, _, _, gui, _ = freqevalNN(Pval[k, :], gui, valrect) 
            fig = showgui(gui) 

            # Créer une figure matplotlib
            # fig, ax = plt.subplots()
            # x = np.linspace(0, 10, 100)
            # y = np.sin(x + i * 0.1)  # Générer une courbe changeante
            # ax.plot(x, y)
            # ax.set_title(f'Frame {i}')
            # ax.set_xlabel('X-axis')
            # ax.set_ylabel('Y-axis')
        
            # Convertir la figure matplotlib en une image numpy
            canvas = FigureCanvas(fig)
            canvas.draw()  # Dessiner la figure sur le canvas
            img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')  # Récupère l'image sous forme RGBA
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Redimensionne en RGBA
            img = img[:, :, :3]  # Convertir en RGB en retirant le canal alpha

            # Fermer la figure pour économiser de la mémoire
            plt.close(fig)

            # Initialiser l'objet VideoWriter une fois avec la taille correcte
            if video is None:
                height, width, _ = img.shape
                video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

            # Convertir l'image matplotlib (RGB) en BGR pour OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
            # Ajouter l'image à la vidéo
            video.write(img_bgr)

    # Libérer l'objet VideoWriter
    # video.release() # enlevé le 15/05/2025

def playVLAM(output_gif,fps, gui, Pval, valrect):
    frames = []
 
    for k in range(len(Pval)):
            # if k % 2 == 0:   # and boolvid:
            _, _, _, gui, _ = freqevalNN(Pval[k, :], gui, valrect) 
            fig = showgui(gui) 
        
            # Convertir la figure matplotlib en une image numpy
            canvas = FigureCanvas(fig)
            canvas.draw()  # Dessiner la figure sur le canvas
            img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')  # Récupère l'image sous forme RGBA
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Redimensionne en RGBA
            img = img[:, :, :3]  # Convertir en RGB en retirant le canal alpha

            # Fermer la figure pour économiser de la mémoire
            plt.close(fig)
                
            frames.append(Image.fromarray(img))

    # Sauvegarder en GIF
    frames[0].save(
        'output.gif',
        save_all=True,
        append_images=frames[1:],
        duration= 5,  # Durée par frame en ms
        loop=0  # Boucle infinie
    )

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
    h1.set_color('g')  # Changer la couleur de la ligne en vert
    h1.set_linewidth(2)  # Épaisseur de la ligne

    # Ajout d'un texte à l'affichage
    ax.text(-0.5, -7.5, 'VLAM Display/GIPSA-LAB', fontsize=18)

    # Définition des limites de l'axe et de l'aspect
    ax.axis([ -1.5, 8.5, -7.5, 2.5])
    ax.set_aspect('equal', adjustable='box')
    
    # Suppression des axes
    ax.axis('off')
    
    return fig  # Retourne le pointeur sur l'objet tracé

def close_on_enter(event):
    if event.key == 'enter':
        plt.close(event.canvas.figure)  # Ferme uniquement la figure associée

# Fonction pour estimer les formants avec l'analyse LPC
def lpc_formants(y, sr, order=16):

    # Appliquer LPC (Linear Predictive Coding)
    A = librosa.lpc(y, order=order)
    
    # Calcul des racines du polynôme LPC
    rts = np.roots(A)
    
    # Garder les racines complexes
    rts = [r for r in rts if np.imag(r) >= 0]
    
    # Convertir en fréquences
    angz = np.angle(rts)
    frqs = angz * (sr / (2 * np.pi))

    # Garder les fréquences inférieures à 5000 Hz (filtrage des fréquences trop hautes)
    formants = sorted(frqs)
    return formants

# Extraire les formants à intervalles réguliers
def extract_formants(y, S, sr, frame_length, hop_length):
    formant_frequencies = []
    num_frames = S.shape[1]  # Nombre de cadres dans le STFT
    for i in range(num_frames):
        start = i * hop_length  # Début de la fenêtre en échantillons
        frame = y[start:start + frame_length]
        if len(frame) == frame_length:
            formants = lpc_formants(frame, sr)
            formant_frequencies.append(formants[:3])  # Prendre les trois premiers formants

    # Convertir la liste de formants en tableau numpy pour pouvoir appliquer le filtrage
    formant_frequencies = np.array(formant_frequencies)

    # Appliquer le filtrage médian sur chaque formant
    filtered_formants = np.zeros_like(formant_frequencies)
    for i in range(formant_frequencies.shape[1]):  # Appliquer pour chaque colonne (chaque formant)
            filtered_formants[:, i] = medfilt(formant_frequencies[:, i], kernel_size=5)  # Taille du filtre médian (peut être ajustée)

    # Créer un vecteur de temps aligné avec le STFT
    times = librosa.frames_to_time(np.arange(num_frames), sr=sr, hop_length=hop_length)
    return times, filtered_formants

def plot_Pv(Pv, 
            labels=('J', 'B', 'D', 'T', 'LP', 'LH', 'Hy'),
            x_label='Frame index',
            y_label='Parameter value',
            title='Evolution of articulatory parameters (Pv)',
            x_range=None,
            y_range=None,
            grid=True,
            fig_number=None):
    """
    Trace les colonnes de Pv en fonction du temps.

    Arguments :
    -----------
    Pv : ndarray de forme (T, N)
        Tableau dont chaque colonne est une série à tracer.
    labels : tuple de str, longueur N
        Noms des séries pour la légende.
    x_label : str
        Étiquette de l’axe des abscisses.
    y_label : str
        Étiquette de l’axe des ordonnées.
    title : str
        Titre du graphique.
    x_range : tuple (xmin, xmax) ou None
        Bornes de l’axe X. Par défaut (0, T-1).
    y_range : tuple (ymin, ymax) ou None
        Bornes de l’axe Y. Par défaut (-4, 5).
    grid : bool
        Affiche la grille si True.
    """
    T, N = Pv.shape
    t = np.arange(T)

    # Crée ou passe sur la figure demandée
    fig = plt.figure(fig_number) if fig_number is not None else plt.figure()

    # Connexion de l'événement clavier à la fonction
    fig.canvas.mpl_connect('key_press_event', close_on_enter)
        
    # plt.figure()
    plt.plot(t, Pv)
    
    # Bornes des axes
    xmin, xmax = x_range if x_range is not None else (0, T-1)
    ymin, ymax = y_range if y_range is not None else (-4, 5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    # Légende sur une seule ligne
    plt.legend(labels, loc='best', ncol=N, frameon=False)
    
    # Titres et style
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if grid:
        plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

def plot_formants(fval,
                  labels=('F1', 'F2', 'F3'),
                  x_label='Frame index',
                  y_label='Formant frequency (Hz)',
                  title='Evolution of Formant Frequencies',
                  x_range=None,
                  y_min=0,
                  top_margin=0.2,
                  grid=True,
                  fig_number=None):
    """
    Trace les formants F1, F2, F3 en fonction du temps,

    Args:
        fval       : ndarray shape (T,3), (T,3,1) ou (3,T)
        labels     : tuple de 3 strings pour la légende
        x_label    : étiquette axe X
        y_label    : étiquette axe Y
        title      : titre du graphique
        x_range    : (xmin, xmax) ou None
        y_min      : valeur minimale de Y
        top_margin : fraction de marge au-dessus du max (ex. 0.2 = 20%)
        grid       : afficher la grille
        fig_number : numéro de figure (int) ou None
    """
   
    # Crée / active la figure
    fig = plt.figure(fig_number) if fig_number is not None else plt.figure()

   # Connexion de l'événement clavier à la fonction
    fig.canvas.mpl_connect('key_press_event', close_on_enter)

    # Prépare fval en un array (T,3)
    arr = np.squeeze(fval)
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"fval must be shape (T,3) or (3,T); got {arr.shape}")

    T = arr.shape[0]
    t = np.arange(T)

    # Trace les trois formants
    plt.plot(t, arr)

    # Bornes X
    xmin, xmax = x_range if x_range is not None else (0, T-1)
    plt.xlim(xmin, xmax)

    # Bornes Y avec marge supplémentaire
    ymax_data = np.nanmax(arr)
    ymin = y_min
    ymax = ymax_data + (ymax_data - ymin) * top_margin
    plt.ylim(ymin, ymax)

    # Légende à l'intérieur, en haut, centrée au-dessus du pic des courbes
    plt.legend(
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95),  # 0.85 = 85% de la hauteur de l'axe
        ncol=3,
        frameon=True,
        borderpad=0.5
    )

    # Étiquettes et style
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if grid:
        plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)

def spectreplot(y, sr, nb, Rs, boolmel):

    frame_length = 400 # int(0.02 * sr)  # 20 ms
    hop_length = 200  # int(0.01 * sr)    # 10 ms (chevauchement de 50%)
    
    if boolmel:    
        # Calculer le spectrogramme Mel
        n_mels = 64           # Nombre de bandes Mel
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels)
    else:
        # Calculer le spectrogramme à l'aide de la transformée de Fourier à court terme (STFT)
        S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

    # Convertir l'amplitude en échelle logarithmique (décibels)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Afficher le spectrogramme
    fig = plt.figure(figsize=(nb*3, 3))
    fig.canvas.mpl_connect('key_press_event', close_on_enter)

    if boolmel:
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel') 
    else:
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear') 
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(Rs)
    plt.xlabel('Temps (s)')
    plt.ylabel('Fréquence (Hz)')
    plt.ylim(0, 4500)
    
    if 0:
        times, filtered_formants = extract_formants(y, S, sr, frame_length, hop_length)
        # Superposer les formants sur le spectrogramme
        for i, formants in enumerate(filtered_formants):
            for f in formants:
                plt.scatter(times[i], f, color='white', s=1)

    plt.tight_layout()
    plt.show(block=True)

    # Ajouter un bouton "Play" pour lire l'audio
    # display(Audio(y, rate=sr))

# Position des consonnes
def consvalD(thetavoy):
    return -4.5 * np.pi / 8
    
def consvalG(thetavoy):
    if thetavoy <= np.pi:    # < /ga/ palatal
        return [1.2, np.pi / 3]
    else:
        return [1.1, -np.pi / 12]
    
def consvalD2(thetavoy):
    return -2.5 * np.pi / 6

def consvalG2():
    return -np.pi / 12

def parse(C, tabvoy, tabcons, rho, theta):
    """
    C       : liste de syllabes (strings)
    tabvoy  : liste/des caractères des voyelles
    tabcons : liste/des caractères des consonnes
    rho     : liste des valeurs ρ pour chaque voyelle
    theta   : liste des valeurs θ pour chaque voyelle
    
    Retour :
      booldeb : liste d’indicateurs pour la position de début (1=consonne, 0=voyelle, -1=aucune voyelle)
      tabdeb  : liste des θ de la 1re voyelle de chaque syllabe (ou -1)
      boolast : liste d’indicateurs pour la position de fin (1=consonne, 0=voyelle, -1=aucune voyelle)
      tablast : liste de [θ, ρ] de la dernière voyelle de chaque syllabe (ou [-1,-1])
    """
    booldeb = []
    tabdeb  = []
    boolast = []
    tablast = []

    for R in C:
        typ = list(R)
        L = len(R)

        # repérage des consonnes
        for k, cons in enumerate(tabcons):
            for i, ch in enumerate(R):
                if ch == cons:
                    typ[i] = cons.upper()   # marque la consonne
        # repérage des voyelles
        for k, voy in enumerate(tabvoy):
            for i, ch in enumerate(R):
                if ch == voy:
                    typ[i] = 'V'            # marque la voyelle

        # positions des voyelles dans typ
        locvoy = [i for i, ch in enumerate(typ) if ch == 'V']

        if locvoy:
            first, last = locvoy[0], locvoy[-1]
            # pour tabdeb et booldeb
            tabdeb.append(theta[tabvoy.index(R[first])])
            booldeb.append(0 if typ[0] == 'V' else 1)
            # pour tablast et boolast
            tablast.append([theta[tabvoy.index(R[last])],
                            rho[tabvoy.index(R[last])]])
            boolast.append(0 if typ[-1] == 'V' else 1)
        else:
            # pas de voyelle trouvée
            booldeb.append(-1)
            tabdeb.append(-1)
            boolast.append(-1)
            tablast.append([-1, -1])

    return booldeb, tabdeb, boolast, tablast

def main():

    # Ajuster la taille de la console Windows
    os.system('mode con: cols=60 lines=5')  # 120 colonnes, 30 lignes

    # Récupérer le handle de la fenêtre de la console
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()

    # Définir les coordonnées et la taille (x, y, largeur, hauteur)
    ctypes.windll.user32.MoveWindow(hwnd, 0, 0, 800, 200, True)  # Positionner en haut de l'écran

    
    plt.close('all')
 
 
    boolvid = 1 # 0 to suppress video output, which slows down the process
    valrect = 0.75 # soft rectification
    Pexp = 2 # Tau model

    # Paramètres du modèle
    co = np.array([[-1.5, 0, np.pi],
                   [-2.5, 0, -np.pi/3],
                   [3, 0, np.pi/3],
                   [-2.75, 0.5, np.pi],
                   [3, 0, np.pi/3],
                   [2.5, 0.5, np.pi],
                   [-2, 0, np.pi/3]])

    gui = initVLAMLength(np.zeros(7), 195)
    cf0 = 1

    dur = 16
    K = 1000 # 10
    Kvoy = 1000 # 30 
    Kpause = 1000 # 30
    nu = -1 

    # Position des voyelles
    theta = [np.pi/3, np.pi/2, 2*np.pi/3, np.pi, 4*np.pi/3, 3*np.pi/2, 5*np.pi/3, 5.5*np.pi/3, 5.5*np.pi/3]
    rho = [1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.7, 0.3] 

    # tabvoy = 'uoOaèéiyE' letters to be changed if needed
    tabvoy = ['u','o','O','a','è','é','i','y','E']
    nbvoy = len(tabvoy)
    # tabcons = 'bdg'
    tabcons = ['b','d','g']
    nbcons = len(tabcons)
    coefcen = 0.5
    art = np.array([[1, 2, 6, 0],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4]])

    art1 = np.array([[1, 2, 3, 6],
                    [1, 2, 3, 1]])
                    
    cons = np.array([[1.2, np.pi/3],   
                     [1.2, 0],
                     [1.1, 0]])

    boolinput = 1
    
    Rs = input('Input words : ')
    boolinput = Rs != 'N'

    while boolinput:
        
        if Rs != '':
            Cw = [word for word in Rs.split()]
            sig = []
            fval = np.array([])
            sylend = [0, 0]
            Pv = np.array([])

            for nword in range(len(Cw)):
                R = Cw[nword]

                C = [word for word in R.split('.')]
                nbsyl = len(C)

                booldeb, tabdeb, boolast, tablast = parse(C, tabvoy, tabcons, rho, theta)
                # tabdeb et tablast non obtenus
                sylcheck = 1 if nbsyl > 1 else 0
                            
                for nsyl in range(nbsyl):
                    R = C[nsyl]
                    typ = list(R)

                    n1 = [-1] * len(typ)
                    for k, conso in enumerate(tabcons):  # tabcons(k) devient cons
                        loccons = [i for i in range(len(R)) if R[i] == conso]  # recherche des positions
                        if loccons:
                            for loc in loccons:
                                typ[loc] = conso.upper()  # conversion en majuscule
                                n1[loc] = k 

                    k1 = [-1] * len(typ)
                    for k, voy in enumerate(tabvoy):  # tabcons(k) devient cons
                        locvoy = [i for i in range(len(R)) if R[i] == voy]  # recherche des positions
                        if locvoy:
                            for loc in locvoy:
                                typ[loc] = 'V'
                                k1[loc] = k 

                    typ = ''.join(typ)
     
                    # on considère « previous » pour la syllabe nsyl-1
                    if sylcheck and nsyl > 0:
                        prev_boolast = boolast[nsyl - 1]
                        curr_booldeb = booldeb[nsyl]
                        prev_tablast = tablast[nsyl - 1]
                        weight = (1 - prev_boolast) + coefcen * prev_boolast * curr_booldeb
                        voydeb = [weight * prev_tablast[1], prev_tablast[0]]
                    else:
                        voydeb = [0.5, tabdeb[nsyl]]

                    print(typ)
                    bool = 1
                    if typ in ['BV', 'DV', 'GV']:
                        cons[1, 1] = consvalD(theta[int(k1[1])])
                        cons[2, :] = consvalG(theta[int(k1[1])])
                        syl = {
                            'pt': np.vstack([voydeb, cons[n1[0]], [rho[k1[1]], theta[k1[1]]]]), 
                            'P': art[n1[0]][art[n1[0]] != 0], 
                            'typ': 'CV'}
                    elif typ in ['BVV', 'DVV', 'GVV']:
                        cons[1, 1] = consvalD(theta[int(k1[1])])
                        cons[2, :] = consvalG(theta[int(k1[1])])
                        syl = {
                            'pt': np.vstack([voydeb, cons[n1[0]], [rho[k1[1]], theta[k1[1]]], [rho[k1[2]], theta[k1[2]]]]), 
                            'P': art[n1[0]][art[n1[0]] != 0], 
                            'typ': 'CVV'}
                    elif typ in ['VBVV', 'VDVV', 'VGVV']:
                        cons[1, 1] = consvalD(theta[int(k1[2])])
                        cons[2, :] = consvalG(theta[int(k1[2])])
                        syl = {
                             'pt': np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]], [rho[k1[2]], theta[k1[2]]], [rho[k1[3]], theta[k1[3]]]]), 
                             'P': art[n1[1]][art[n1[1]] != 0],
                             'typ': 'VCVV'}
                    elif typ in ['VB', 'VD', 'VG']:
                        cons[1, 1] = consvalD(theta[int(k1[0])])
                        cons[2, :] = consvalG(theta[int(k1[0])])
                        syl = {
                            'pt': np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]], [coefcen * rho[k1[0]], theta[k1[0]]]]), 
                            'P': art[n1[1]][art[n1[1]] != 0],
                            'typ': 'VC'}
                    elif typ in ['VVB', 'VVD', 'VVG']:
                        cons[1, 1] = consvalD(theta[int(k1[0])])
                        cons[2, :] = consvalG(theta[int(k1[0])])
                        syl = {
                              'pt': np.vstack([[rho[k1[0]], theta[k1[0]]], [rho[k1[1]], theta[k1[1]]], cons[n1[2]], [coefcen * rho[k1[1]], theta[k1[1]]]]), 
                              'P': art[n1[2]][art[n1[2]] != 0],
                              'typ': 'VVC'}
                    elif typ in ['VBV', 'VDV', 'VGV']:
                        cons[1, 1] = consvalD(theta[int(k1[2])])
                        cons[2, :] = consvalG(theta[int(k1[2])])
                        syl = {
                               'pt': np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]], [rho[k1[2]], theta[k1[2]]]]), 
                               'P': art[n1[1]][art[n1[1]] != 0],
                               'typ': 'VCV'}
                    elif typ in ['BVB', 'BVD', 'BVG', 'DVB', 'DVD', 'DVG', 'GVB', 'GVD', 'GVG']:
                        cons[1][1] = consvalD(theta[k1[1]])
                        cons[2, :] = consvalG(theta[k1[1]])
                        syl = {
                            'pt': np.vstack([voydeb, cons[n1[0]], [rho[k1[1]], theta[k1[1]]], cons[n1[2]], [coefcen * rho[k1[1]], theta[k1[1]]]]),
                            'P1': art[n1[0]][art[n1[0]] != 0],
                            'P2': art[n1[2]][art[n1[2]] != 0],
                            'typ': 'CVC'
                        }
                    elif typ in ['VBVB', 'VBVD', 'VBVG', 'VDVB', 'VDVD', 'VDVG', 'VGVB', 'VGVD', 'VGVG']:
                        cons[1][1] = consvalD(theta[k1[2]])
                        cons[2, :] = consvalG(theta[k1[2]])
                        syl = {
                            'pt': np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]], [rho[k1[2]], theta[k1[2]]], cons[n1[3]], [coefcen * rho[k1[2]], theta[k1[2]]]]),
                            'P1': art[n1[1]][art[n1[1]] != 0], 
                            'P2': art[n1[3]][art[n1[3]] != 0], 
                            'typ': 'VCVC'
                        }
                    elif typ in ['BVVB', 'BVVD', 'BVVG', 'DVVB', 'DVVD', 'DVVG', 'GVVB', 'GVVD', 'GVVG']:
                        cons[1][1] = consvalD(theta[k1[1]])
                        cons[2, :] = consvalG(theta[k1[1]])
                        syl = {
                            'pt': np.vstack([voydeb, cons[n1[0]], [rho[k1[1]], theta[k1[1]]], [rho[k1[2]], theta[k1[2]]], cons[n1[3]], [coefcen * rho[k1[2]], theta[k1[2]]]]),
                            'P1': art[n1[0]][art[n1[0]] != 0],
                            'P2': art[n1[3]][art[n1[3]] != 0],
                            'typ': 'CVVC'
                        }
                    elif typ in ['VBVVB', 'VBVVD', 'VBVVG', 'VDVVB', 'VDVVD', 'VDVVG', 'VGVVB', 'VGVVD', 'VGVVG']:
                        cons[1][1] = consvalD(theta[k1[2]])
                        cons[2, :] = consvalG(theta[k1[2]])
                        syl = {
                            'pt': np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]] ,[rho[k1[2]], theta[k1[2]]], [rho[k1[3]], theta[k1[3]]], cons[n1[4]], [coefcen * rho[k1[3]], theta[k1[3]]]]),
                            'P1': art[n1[1]][art[n1[1]] != 0],
                            'P2': art[n1[4]][art[n1[4]] != 0],
                            'typ': 'VCVVC'
                        }
                    elif typ in ['VBD', 'VDB', 'VBG', 'VGB', 'VDG', 'VGD']:
                        cons[1][1] = consvalD2(theta[k1[0]])
                        cons[2][1] = consvalG2()
                        syl = {
                            'pt': np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]], cons[n1[2]], [coefcen * rho[k1[0]], theta[k1[0]]]]),
                            'P': art1[int(n1[1] + n1[2] > 2)],
                            'typ': 'VcC'
                        }
                    elif typ in ['VVBD', 'VVDB', 'VVBG', 'VVGB', 'VVDG', 'VVGD']:
                        cons[1][1] = consvalD2(theta[k1[1]])
                        cons[2][1] = consvalG2()
                        syl = {
                            'pt': np.vstack([[rho[k1[0]], theta[k1[0]]], [rho[k1[1]], theta[k1[1]]], cons[n1[2]], cons[n1[3]], [coefcen * rho[k1[1]], theta[k1[1]]]]),
                            'P': art1[int(n1[2] + n1[3] > 2)],
                            'typ': 'VVcC'
                        }
                    elif typ in ['BDV', 'DBV', 'BGV', 'GBV', 'DGV', 'GDV']:
                        cons[1][1] = consvalD2(theta[k1[2]])
                        cons[2][1] = consvalG2()
                        syl = {
                            'pt': np.vstack([voydeb, cons[n1[0]], cons[n1[1]], [rho[k1[2]], theta[k1[2]]]]),
                            'P': art1[int(n1[0] + n1[1] > 2)],
                            'typ':  'CcV'
                        }
                    elif typ in ['VBDV', 'VDBV', 'VBGV', 'VGBV', 'VDGV', 'VGDV']:
                        cons[1][1] = consvalD2(theta[k1[3]])
                        cons[2][1] = consvalG2()
                        syl = {
                            'pt': np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]], cons[n1[2]], [rho[k1[3]], theta[k1[3]]]]),
                            'P': art1[int(n1[1] + n1[2] > 2)],
                            'typ': 'VCcV'
                        }
                    elif typ in ['BDVV', 'DBVV', 'BGVV', 'GBVV', 'DGVV', 'GDVV']:
                        cons[1][1] = consvalD2(theta[k1[2]])
                        cons[2][1] = consvalG2()
                        syl = {
                            'pt': np.vstack([voydeb, cons[n1[0]], cons[n1[1]], [rho[k1[2]], theta[k1[2]]], [rho[k1[3]], theta[k1[3]]]]),
                            'P': art1[int(n1[0] + n1[1] > 2)],
                            'typ':  'CcVV'
                        }
                    elif typ in ['VBDVV', 'VDBVV', 'VBGVV', 'VGBVV', 'VDGVV', 'VGDVV']:
                        cons[1][1] = consvalD2(theta[k1[3]])
                        cons[2][1] = consvalG2()
                        syl = {
                            'pt': np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]], cons[n1[2]], [rho[k1[3]], theta[k1[3]]], [rho[k1[4]], theta[k1[4]]]]),
                            'P': art1[int(n1[1] + n1[2] > 2)],
                            'typ':  'VCcVV'
                         } 
                    else:
                        print('Unknown syllable')
                        bool = 0
                        sylcheck = 0

                    if bool:
                        if sylcheck:
                            Pval = boucleword(syl, co, nu, K, Kvoy, dur, Pexp)
                            Tval = bouclewordplot(syl, nu, K, Kvoy, dur, Pexp)                            

                            if nsyl == 0:
                                word = ['O', syl['typ']]
                                Pvalword = Pval
                                syldeb = syl['pt'][0]
                                sig1 = []
                                fval1 = []
                            elif nsyl == nbsyl-1:
                                sig1, fval1 = synthwordfen(gui, np.vstack([Pvalword, Pval]), word + [syl['typ'], 'F'], cf0, valrect, dur)
                                Pvalpause = np.array(arc(np.vstack([sylend, syldeb]), co, np.arange(0, 7), dur, [-np.pi, 0], 0, nu, Kpause, Pexp))
                                sylend = syl['pt'][-1]
                                if Pv.size == 0:
                                        Pv = np.vstack([Pvalpause, Pvalword, Pval])  # Initialisation si Pv est vide
                                else:
                                        Pv = np.vstack([Pv, np.vstack([Pvalpause, Pvalword, Pval])])  # Empilement si Pv a déjà des données
                                sig = np.concatenate([sig, np.zeros(dur * 200), sig1])
                                if fval.size == 0:
                                        fval = np.hstack([synthpause(gui, Pvalpause, valrect).T, fval1])  # Initialisation si Pv est vide
                                else:
                                        fval = np.hstack([fval, np.hstack([synthpause(gui, Pvalpause, valrect).T, fval1])])  # Empilement si Pv a déjà des données                     
                            else:
                                word.append(syl['typ'])
                                Pvalword = np.vstack([Pvalword, Pval])
                                sig1 = []
                                fval1 = []
                        else:
                            Pval = boucle(syl, co, nu, K, Kvoy, dur, Pexp)
                            Tval = boucleplot(syl, nu, K, Kvoy, dur, Pexp)
                            fig = plt.figure(2)
                            plt.clf()  # Efface les tracés précédents de la figure active
                            fig.canvas.mpl_connect('key_press_event', close_on_enter)
                            plt.polar(np.angle(Tval), np.abs(Tval))
                            plt.polar(np.angle(Tval), np.abs(Tval),'.')
                            plt.title(R, fontsize=22)
                            sig1, fval1 = synthsyl(gui, np.array(Pval), syl, cf0, valrect)
                            Pvalpause = np.array(arc(np.vstack([sylend, syl['pt'][0]]), co, np.arange(0, 7), dur, [-np.pi, 0], 0, nu, Kpause, Pexp))
                            sylend = syl['pt'][-1]
     
                            sig = np.concatenate([sig, np.zeros(dur * 200), sig1])
                            if fval.size == 0:
                                fval = np.hstack([synthpause(gui, Pvalpause, valrect).T, fval1])  # Initialisation si fval est vide
                            else:
                                fval = np.hstack([fval, np.hstack([synthpause(gui, Pvalpause, valrect).T, fval1])])  # Empilement si fval a déjà des données                     
                            if Pv.size == 0:
                                Pv = np.vstack([Pvalpause, Pval])  # Initialisation si Pv est vide
                            else:
                                Pv = np.vstack([Pv, np.vstack([Pvalpause, Pval])])  # Empilement si Pv a déjà des données


            if boolvid:
                playVLAMvid('output.avi', 100, gui, Pv, valrect)

            if 1:
                plot_Pv(Pv,
                    labels=['J','B','D','T','LP','LH','Hy'],
                    x_range=(0, Pv.shape[0]-1),
                    y_range=(-4,5), fig_number=5)
                plot_formants(fval, fig_number=6)
     
            if len(sig) > 0:
                signal_int16 = np.int16(32767 * sig / (1.20 * np.max(np.abs(sig))))
                write("essai.wav", 20000, signal_int16)
                if 1:
                    spectreplot(sig, 20000, len(Cw), Rs, 0)
                np.save('essai.npy', {'fval': fval, 'Pv': Pv, 'dur': dur})
           
            Rs = input('Input words (or N to stop): ').rstrip('\n')  
            boolinput = not (Rs == 'N')
        else:
            Rs = input('Input words (or N to stop): ')  
            boolinput = not (Rs == 'N')
            
if __name__ == "__main__":
        main()
