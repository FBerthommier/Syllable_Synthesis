import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Générer trois nuages de points en 3D
def generate_data_Ohman():   
	# Données extraites du tableau
    datavbv = np.array([
        [-35, -90, -350, -265, -330, -235],
        [-20, -20, -445, -235, -285, -85],
        [-40, -45, -500, -325, -295, -95],
        [-20, -20, -575, -320, -260, -115],
        [-10, -90, -475, -390, -320, -130],

        [-95, -130, -275, -50, -90, -155],
        [-95, -125, -190, -130, -110, -80],
        [-85, -150, -355, -160, -40, -110],
        [-105, -140, -345, -235, -80, -65],
        [-115, -150, -370, -270, -125, -115],

        [-300, -325, 150, 35, -290, -255],
        [-220, -270, 95, -15, -265, -280],
        [-280, -330, -170, -55, -140, -210],
        [-260, -330, -155, -80, -140, -145],
        [-350, -300, -150, -185, -190, -225],

        [-85, -105, 290, 255, np.nan, -300],
        [-120, -140, 255, 80, -250, -150],
        [-110, -105, -25, 60, -50, -55],
        [-140, -95, -5, 0, -10, np.nan],
        [-100, -130, -10, -20, np.nan, -125],

        [-60, -50, 375, 225, -130, -315],
        [-60, -110, 175, 275, np.nan, -135],
        [-90, -80, -135, 170, np.nan, -90],
        [-80, -65, -50, 5, np.nan, np.nan],
        [-85, -90, -120, -25, np.nan, -155]])
	
    # Extraction de la colonne de la voyelle finale F1 F2 F3 en colonne
    datavbvfin = datavbv[:,1::2]
    datavbvinit = datavbv[:,::2]

    datavdv = np.array([
        [-50, -115, -85, -85, 245, 255],
        [-80, -95, -170, -185, 150, 265],
        [-65, -95, -260, -290, 250, 320],
        [-65, -110, -140, -290, 180, 220],
        [-70, -65, -155, -235, 165, 135],

        [-115, -170, -40, 150, 210, 135],
        [-145, -180, -55, 65, 205, 255],
        [-110, -135, -210, -25, 280, 240],
        [-135, -135, -135, -115, 245, 165],
        [-155, -180, -95, 15, 290, 105],

        [-335, -355, 365, 440, -95, -20],
        [-385, -390, 310, 480, -95, -50],
        [-320, -350, 125, 225, -65, -50],
        [-385, -270, 175, 170, -20, -75],
        [-365, -375, 280, 220, -125, -65],

        [-80, -215, 590, 870, 50, 100],
        [-85, -160, 580, 710, np.nan, 50],
        [-50, -200, 330, 500, np.nan, 120],
        [-75, -95, 365, 475, np.nan, 40],
        [-110, -175, 425, 520, np.nan, 30],

        [-65, -125, 665, 900, np.nan, -50],
        [-70, -110, 665, 800, np.nan, np.nan],
        [-60, -45, 345, 700, np.nan, np.nan],
        [-25, -35, 430, 650, np.nan, np.nan],
        [-60, -95, 570, 620, np.nan, -50]])

	
    datavdvfin = datavdv[:,1::2]
    datavdvinit = datavdv[:,::2]
    
    datavgv = np.array([
        [-60, -75, 130, 145, -195, -120],
        [-80, -125, 95, 185, -220, -15],
        [-65, -70, -185, 110, -300, -35],
        [-10, -55, -410, 190, -310, -125],
        [-40, -55, -375, 185, -245, -25],

        [-130, -215, 230, 360, -300, -185],
        [-130, -175, 115, 300, -205, -195],
        [-115, -150, -130, 410, -155, -140],
        [-95, -130, -185, 300, -180, -215],
        [-90, -140, -240, 340, -150, -175],

        [-320, -420, 510, 430, -315, -355],
        [-335, -370, 400, 405, -425, -275],
        [-270, -325, 225, 110, -275, -145],
        [-275, -355, 50, 55, -200, -25],
        [-310, -300, 110, 20, -130, -80],

        [-100, -175, 195, 260, np.nan, np.nan],
        [-105, -125, 190, 165, np.nan, -190],
        [-115, -140, 110, 110, np.nan, np.nan],
        [-120, -135, 35, 10, np.nan, np.nan],
        [-105, -120, 15, -25, np.nan, np.nan],

        [-65, -135, 105, 280, np.nan, -265],
        [-55, -125, 60, 180, np.nan, -235],
        [-75, -50, -60, 135, np.nan, np.nan],
        [-100, -95, -65, 40, np.nan, np.nan],
        [-70, -95, -120, -50, np.nan, np.nan]])
   
    datavgvfin = datavgv[:,1::2]
    datavgvinit = datavgv[:,::2]

    formantvbv = np.array([ 
        [290, 330, 400, 430, 670, 660, 400, 420, 340, 360],
        [2000, 1890, 1650, 1580, 990, 1040, 670, 730, 670, 680],
        [2350, 2320, 2320, 2380, 2670, 2620, np.nan, 2450, np.nan, 2430],
    ])
    
    fvbvinit = formantvbv[:, ::2]
    fvbvfin = formantvbv[:,1::2]

    formantvdv = np.array([ 
         [270, 300, 390, 400, 660, 650, 390, 410, 330, 320],
         [1990, 1950, 1620, 1580, 970, 1030, 690, 760, 660, 720],
         [2380, 2350, 2350, 2410, 2740, 2700, 2530, 2560, np.nan, 2500],
    ])
    
    fvdvinit = formantvdv[:, ::2]
    fvdvfin = formantvdv[:,1::2]

    formantvgv = np.array([ 
         [320, 330, 420, 420, 660, 650, 430, 430, 360, 360],
         [2010, 1960, 1650, 1650, 990, 1160, 780, 840, 770, 790],
         [2440, 2340, 2440, 2350, 2560, 2480, 2560, 2450, 2480, 2460],
    ])
    
    fvgvinit = formantvgv[:,::2]
    fvgvfin = formantvgv[:,1::2]
    
    # # Conversion en tableau NumPy
    # datavgv_array = np.array(datavgv, dtype=np.float64)

    # # Affichage du tableau NumPy
    # print("Tableau NumPy datavgv :")
    # print(datavgv_array)

    # # Exemple de traitement : affichage des valeurs non manquantes
    # print("\nValeurs non manquantes :")
    # print(datavgv_array[~np.isnan(datavgv_array)])
    
    tabvbvfc= np.zeros([25,3])
    tabvdvfc = np.zeros([25,3])
    tabvgvfc = np.zeros([25,3])
    tabvbvfv= np.zeros([25,3])
    tabvdvfv = np.zeros([25,3])
    tabvgvfv = np.zeros([25,3])

    # Par blocs de 5 on additionne fvxvfin à datavxvfin ligne par colonne pour F1 F2 F3
    
    for i in range(5):
        for j in range (5):
            for k in range(3):
                tabvbvfc[i*5+j,k] = datavbvfin[i*5+j,k] + fvbvfin[k,i]
                tabvdvfc[i*5+j,k] = datavdvfin[i*5+j,k] + fvdvfin[k,i]
                tabvgvfc[i*5+j,k] = datavgvfin[i*5+j,k] + fvgvfin[k,i]
                tabvbvfv[i*5+j,k] = fvbvfin[k,i]
                tabvdvfv[i*5+j,k] = fvdvfin[k,i]
                tabvgvfv[i*5+j,k] = fvgvfin[k,i]
  
    cloud1 = np.vstack([tabvbvfv[:, 1], tabvbvfc[:, 1], tabvbvfc[:, 2]]).T
    cloud2 = np.vstack([tabvdvfv[:, 1], tabvdvfc[:, 1], tabvdvfc[:, 2]]).T
    cloud3 = np.vstack([tabvgvfv[:, 1], tabvgvfc[:, 1], tabvgvfc[:, 2]]).T
    labels1 = np.zeros(len(cloud1))
    labels2 = np.ones(len(cloud2))
    labels3 = np.full(len(cloud3), 2)
    print(cloud1)
    return (
        np.vstack([cloud1, cloud2, cloud3]),
        np.hstack([labels1, labels2, labels3]),
    )

def generate_data():
    # Charger le fichier .npy
    data = np.load('LocusOhman.npy', allow_pickle=True).item()  # .item() pour récupérer le dictionnaire

    # Accéder aux variables
    fval = np.squeeze(data['fval']).T  # Variable contenant des valeurs (par exemple des fréquences ou des instants)
    # Pv = data['Pv']      # Variable associée (par exemple des puissances, valeurs observées)
    dur = data['dur']    # Variable temporelle ou durée totale


    # Vérifier les dimensions de fval
    print("Dimensions de fval :", fval.shape)

    # Paramètres
    T = dur             # Exemple de valeur pour T (à définir selon votre besoin)
    decal = 4          # Exemple de valeur pour le décalage
    nb = 75  # Nombre de valeurs
    
    # Locus CV
    ind1 = np.arange(decal+3*T, 6000, 5*T)
    ind2 = np.arange(4*T, 6000, 5*T)

    # Locus VC
    # ind1 = np.arange(3*T-decal, 6000, 5*T)
    # ind2 = np.arange(2*T, 6000, 5*T)
    
    f2c=np.zeros([25,3])
    f3c=np.zeros([25,3])
    f2v=np.zeros([25, 1])
    f2c[:, 0]=fval[ind1[0:25],1]
    f2c[:, 1]=fval[ind1[25:50],1]
    f2c[:, 2]=fval[ind1[50:75],1]
    f3c[:, 0]=fval[ind1[0:25],2]
    f3c[:, 1]=fval[ind1[25:50],2]
    f3c[:, 2]=fval[ind1[50:75],2]
    f2v=fval[ind2[0:25],1]
  
    cloud1 = np.vstack([f2v, f2c[:, 0], f3c[:, 0]]).T
    cloud2 = np.vstack([f2v, f2c[:, 1], f3c[:, 1]]).T
    cloud3 = np.vstack([f2v, f2c[:, 2], f3c[:, 2]]).T
    labels1 = np.zeros(len(cloud1))
    labels2 = np.ones(len(cloud2))
    labels3 = np.full(len(cloud3), 2)
    print(cloud1)
    return (
        np.vstack([cloud1, cloud2, cloud3]),
        np.hstack([labels1, labels2, labels3]),
    )
     
def plot_convex_hull_3d(points, ax, color='blue', alpha=0.3):
    """
    Trace un Hull convexe en 3D pour un ensemble de points.

    Parameters:
        points (np.ndarray): Tableau de points (n_points, 3).
        ax (Axes3D): Objet matplotlib Axes3D pour l'affichage.
        color (str): Couleur du Hull.
        alpha (float): Transparence du Hull.
    """
    try:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            triangle = points[simplex]
            poly = Poly3DCollection([triangle], alpha=alpha, color=color, edgecolor='k')
            ax.add_collection3d(poly)
        # print(f"Hull tracé avec {len(hull.simplices)} faces.")
    except Exception as e:
        print(f"Erreur lors du tracé du Hull : {e}")

def preprocess_points(points):
    """
    Supprime les points contenant des NaN.
    """
    return points[~np.isnan(points).any(axis=1)]

def close_on_enter(event):
    if event.key == 'enter':
        plt.close(event.canvas.figure)  # Ferme uniquement la figure associée

def visualize_3d_with_alpha_shapes(X, y, labels, boolS, alpha):
    """
    Visualise des alpha shapes pour plusieurs classes en 3D.

    Parameters:
        X (np.ndarray): Points de données, forme (n_samples, 3).
        y (np.ndarray): Étiquettes de classe, forme (n_samples,).
        labels (list): Liste des classes uniques.
        alpha (float): Paramètre alpha pour calculer les alpha shapes.
    """
    fig = plt.figure(boolS+1, figsize=(10, 7))
    fig.canvas.mpl_connect('key_press_event', close_on_enter)
    ax = fig.add_subplot(111, projection="3d")

    if boolS: 
        # ax.set_xlim(2700, 700)
        ax.set_xlim(700, 2700)
        ax.set_ylim(700, 2700)
        ax.set_zlim(1800, 4000)
    else:
        # ax.set_xlim(2500, 400)
        ax.set_xlim(400, 2500)
        ax.set_ylim(400, 2500)
        ax.set_zlim(1850, 3000)
 
    clist = ["b", "d", "g"]
    colors = ['b', 'g', 'r']
    
    for i, label in enumerate(labels):
        class_points = X[y == label]
        class_points = preprocess_points(class_points) 
        ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2],
                   label=f"Class {label}", color=colors[i % len(colors)], alpha=0.6)
        
        if len(class_points) >= 4:  # Besoin de 4 points minimum en 3D
            plot_convex_hull_3d(class_points, ax, color=colors[i % len(colors)], alpha=0.3)
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Maximum de 5 ticks principaux
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Maximum de 4 ticks principaux
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))  # Maximum de 4 ticks principaux
    
    ax.set_xlabel('F2v (Hz)')
    ax.set_ylabel('F2c (Hz)')
    ax.set_zlabel('F3c (Hz)')
    
    if boolS:
        ax.set_title('Synthetic data')
    else:
        ax.set_title('Öhman 1966 spectrographic measurements')
    
    # Réglage de la vue initiale
    ax.view_init(elev=22, azim=-92)

    # ax.legend()

    plt.show(block=True)


if __name__ == "__main__":
    # boolS= 0 = Ohman # 1 = Synthetic
    
    labels = [0, 1, 2]
    X2, y2 = generate_data_Ohman()    
    print("Press Enter for next figure")
    visualize_3d_with_alpha_shapes(X2, y2, labels, 0, alpha=1.0)
    
    X1, y1 = generate_data()
    visualize_3d_with_alpha_shapes(X1, y1, labels, 1, alpha=1.0)

 