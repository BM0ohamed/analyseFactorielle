#import codage as cd
import numpy as np
from numpy import *
from matplotlib.pyplot import *
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.vq import whiten,kmeans2


def normalisation(mat):
    moy=mat.mean(0) #on obtient la moyenne par colonne
    ectype = mat.std(0) #on obtient l'écart type par colonne
    mat_norm = (mat - moy) / ectype
    return mat_norm


def qttitaf_en_quali_1(mat,nb_intervalle):
    [nb_ligne,nb_colone]=np.shape(mat)
    mmin=mat.min(0) #liste du minimum de chaque colonne, pour le min de chaque ligne mat.min(1)
    mmax=mat.max(0) #le max de chaque colonne
    taille_intervalles = (mmax-mmin)/nb_intervalle
    bornes=mmin
    for i in range(1,nb_intervalle):
        bornes=np.vstack((bornes,i*taille_intervalles + mmin))
    bornes = np.vstack((bornes,mmax))

    mat_res = np.zeros(mat.shape,dtype='int') #on initialise la matrice

    for i in range(nb_ligne):
        for j in range(nb_colone):
            k=0
            trouve=False
            while not trouve:
                if bornes[k,j] <= mat[i,j] <= bornes[k+1,j]:
                    trouve= True
                    mat_res[i,j]=k
                k+=1
    return mat_res

#%%
#===========================================================================================================================#
#=============================-Cette méthode correspond à un mix de ACP + CAH-==============================================#

#Dans cette méthode on utilise l'ACP pour déterminer les facteurs des individus puis on utilisera ce vecteur pour la CAH
def premiereMethode(X,nomIndividus,nomVariables,k):
    #On commence par normaliser les données
    X = normalisation(X)
    #On effectue l'ACP
    print("==================================================")
    print("Lancement de la première méthode : ACP + CAH")
    [nb_individus,nb_variables]=np.shape(X)
    D=1/nb_individus *np.identity(nb_individus)
    M=np.identity(nb_variables) # M matrice métrique qui permet d'avoir le meme poids sur toutes les variables
    res={'matD' : D, 'matM' : M}

    Xcov_ind=X.T.dot(D.dot(X.dot(M)))
    L,U=np.linalg.eig(Xcov_ind)
    
    indices=np.argsort(L)[: :-1]
    val_p_ind=np.sort(L)[: :-1]
    vect_p_ind=U[:,indices]


    Xcov_var=X.dot(M.dot(X.T.dot(D)))
    L,U=np.linalg.eig(Xcov_var)

    val_p_var=np.sort(L)[: :-1]

    fact_ind= X.dot(M.dot(vect_p_ind)) #Fs=X.M.u_s
    fact_var = X.T.dot(D.dot(fact_ind)) / sqrt(val_p_ind) # Gs = 1/sqrt(lambda) * X' D Fs
    
    contribution_ind= np.zeros(fact_ind.shape)

    for i in range(fact_ind.shape[1]):
        f=fact_ind[:,i] #f correspond à la colonne
        contribution_ind[:,i]=100*D.dot(f*f) / f.T.dot(D.dot(f))

    distance= (fact_ind **2).sum(1).reshape(fact_ind.shape[0],1)
    qualite_ind = 100*(fact_ind**2) / distance 

    inerties = 100* val_p_ind / val_p_ind.sum()
 
    #diagramem des inerties 
    fig1=figure(figsize=(6, 4))
    title('Diagramme des inerties')
    xlabel('Composantes')
    ylabel('Pourcentage d inerties')
    plot(inerties)
    show()
    
    #On affiche les individus sur le plan factoriel
    fig2=figure(figsize=(6, 4))
    title('Individus sur le plan factoriel')
    xlabel('F1')
    ylabel('F2')
    plot(fact_ind[:,0],fact_ind[:,1],'o')
    #afficher l'axe 0,0
    plot([0,0],[fact_ind[:,1].min(),fact_ind[:,1].max()],'k')
    plot([fact_ind[:,0].min(),fact_ind[:,0].max()],[0,0],'k')
    for i in range(nb_individus):
        text(fact_ind[i,0],fact_ind[i,1],nomIndividus[i])
    show()
    
    # Affichage du plan factoriel pour les variables actives
    fig3 = figure(figsize=(6, 4))
    x = np.arange(-1,1,0.001)
    cercle_unite = np.zeros((2,len(x)))
    cercle_unite[0,:] = np.sqrt(1-x**2)
    cercle_unite[1,:] = -cercle_unite[0,:]
    plot(x,cercle_unite[0,:])
    plot(x,cercle_unite[1,:])
    plot(fact_var[:,0],fact_var[:,1],'x')
    yscale('linear')
#    ylim(-1.2,1.2)
#    xlim(-1.2,1.2)
    grid(True)
    axvline(linewidth=0.5,color='k')
    axhline(linewidth=0.5,color='k')
    title('ACP Projection des variables')
    
    for label,x,y in zip(nomVariables,fact_var[:,0],fact_var[:,1]):
        annotate(label,
                 xy = (x,y),
                 xytext = (-50,5),
                 textcoords = 'offset points',
                 arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0')
                 )
    
    somme=0
    compteurAxes=0
    while(somme<90):
        somme+=inerties[compteurAxes]
        compteurAxes+=1

    #On effectue la CAH
    Z = linkage(fact_ind[:,:compteurAxes],method='ward',metric='euclidean')
    #On affiche le dendrogramme
    fig1=figure(figsize=(6, 4))
    title('CAH')
    xlabel('Individus')
    ylabel('Distance')
    dendrogram(
        Z,
        labels=nomIndividus,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    show()
    #On effectue le clustering
    cluster = fcluster(Z,k,criterion='maxclust')
    #On affiche les individus par cluster
    for i in range(1,k+1):
        print("Individus du cluster ",i,":")
        for j in range(len(cluster)):
            if cluster[j]==i:
                print(nomIndividus[j])
        print("")
    
    #Affichage du nuage de points des individus par cluster
    fig4=figure(figsize=(6, 4))
    title('Individus par cluster')
    xlabel('F1')
    ylabel('F2')
    for i in range(1,k+1):
        plot(fact_ind[cluster==i,0],fact_ind[cluster==i,1],'o')
    #afficher l'axe 0,0
    plot([0,0],[fact_ind[:,1].min(),fact_ind[:,1].max()],'k')
    plot([fact_ind[:,0].min(),fact_ind[:,0].max()],[0,0],'k')
    for i in range(nb_individus):
        text(fact_ind[i,0],fact_ind[i,1],nomIndividus[i])
    show()

    print("fin de la première méthode : ACP + CAH")
    print("==================================================")
    

    return None



#%%
#===========================================================================================================================#
#=============================-Cette méthode correspond à un mix de ACP + Centre mobiles-===================================#


def methode_mixte2(X,nom_individus,nom_variables,k):
# Je décide de mettre k en paramètre pour le modifier plus facilement.
# k est le nombre de groupes que fournit la classification non supervisée
    print("==================================================")
    print("Lancement de la deuxième méthode : ACP + Centres mobiles")
    X = normalisation(X)
    
    I,K = X.shape[0],X.shape[1]
    
# Calcul du vecteur fact_ind

    #Calcul de la matrice M (métrique), D (poid des individus)
    M = identity(K)
    D = (1/I)*identity(I)
    
    # Tri par ordre décroissant des valeurs des valeurs propres de la matrice de covariance
    # Calcul de la matrice de covariance pour les individus
    X_cov_ind = X.T.dot(D.dot(X.dot(M)))
    L,U = linalg.eig(X_cov_ind)
    
    indices = argsort(L)[::-1]
    val_p_ind = sort(L)[::-1]
    vect_p_ind = U[:,indices]
    
    # Calcul des facteurs pour les individus 
    fact_ind = X.dot(M.dot(vect_p_ind)) # Fs = X.M.u_s
    
     # Calcul des facteurs pour les variables actives (utilisation des relations de transition) 
    fact_var = X.T.dot(D.dot(fact_ind)) / sqrt(val_p_ind) # Gs = 1/sqrt(lambda) * X' D Fs


# Centres mobiles à partir du vecteur fact_ind
    centre, labels = kmeans2(fact_ind,k)

    #On affiche le diagramme des inerties
    fig1=figure(figsize=(6, 4))
    title('Diagramme des inerties')
    xlabel('Composantes')
    ylabel('Pourcentage d inerties')
    plot(val_p_ind/val_p_ind.sum()*100)
    show()

    # Affichage du plan factoriel pour les variables actives
    fig3 = figure(figsize=(6, 4))
    x = np.arange(-1,1,0.001)
    cercle_unite = np.zeros((2,len(x)))
    cercle_unite[0,:] = np.sqrt(1-x**2)
    cercle_unite[1,:] = -cercle_unite[0,:]
    plot(x,cercle_unite[0,:])
    plot(x,cercle_unite[1,:])
    plot(fact_var[:,0],fact_var[:,1],'x')
    yscale('linear')
#    ylim(-1.2,1.2)
#    xlim(-1.2,1.2)
    grid(True)
    axvline(linewidth=0.5,color='k')
    axhline(linewidth=0.5,color='k')
    title('ACP Projection des variables')
    
    for label,x,y in zip(nom_variables,fact_var[:,0],fact_var[:,1]):
        annotate(label,
                 xy = (x,y),
                 xytext = (-50,5),
                 textcoords = 'offset points',
                 arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0')
                 )
    
# Tracer des données 
    figure(figsize=(10, 8))
    scatter(fact_ind[:,0], fact_ind[:,1], c=labels)
    for label,x,y in zip(nom_individus,fact_ind[:,0],fact_ind[:,1]):
        annotate(label,
            xy=(x,y),
		    xytext=(-50,5),
		    textcoords='offset points',
		    ha='right', va='bottom',
		    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
		    )
    title('Combinaison méthode ACP + Centres mobiles')
    grid(True)
    axvline(linewidth=0.5,color='k')
    axhline(linewidth=0.5,color='k')
    

    show()

    print("Fin de la deuxième méthode : ACP + Centres mobiles")
    print("==================================================")


#%%
#===========================================================================================================================#
#=============================-Cette méthode correspond à un mix de ACM + CAH-==============================================#

#ACM et CAH
def troisiemeMehode(X,nom_individus,nom_variables,k):
    X=normalisation(X)
    #On effectue l'ACM
    print("==================================================")
    print("Lancement de la troisième méthode : ACM + CAH")
    [nb_ligne,nb_colone]=np.shape(data)
    X=qttitaf_en_quali_1(X,5)
    nb_mod_par_var = data.max(0)
    nb_mod=int(nb_mod_par_var.sum())

    XTDC = np.zeros((data.shape[0],nb_mod),dtype="int")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            XTDC[i,int(data[i,j]-1 + nb_mod_par_var[:j].sum())]=1

    # print("tableau de burt :")
    # print(mat)
    mat=XTDC

    Xfreq = mat / mat.sum()
    marge_colonne = Xfreq.sum(1).reshape(Xfreq.shape[0],1)
    marge_ligne = Xfreq.sum(0).reshape(1,Xfreq.shape[1])

    Xindep = marge_ligne * marge_colonne
    X = Xfreq/Xindep - 1
    M = np.diag(marge_ligne[0,:])
    D = np.diag(marge_colonne[:,0])


    Xcov_ind=X.T.dot(D.dot(X.dot(M)))

    L,U=np.linalg.eig(Xcov_ind) #récupere les valeurs propres et les vecteurs propres
    L=np.float16(L)
    U=np.float16(U)
    
    indices=np.argsort(L)[: :-1]
    val_p_ind=np.sort(L)[: :-1]
    vect_p_ind=U[:,indices]

    Xcov_var=X.dot(M.dot(X.T.dot(D)))
    L,U=np.linalg.eig(Xcov_var)

    #tri par ordre décroissant des valeurs propres
    val_p_var=np.sort(L)[: :-1]


    fact_ind= X.dot(M.dot(vect_p_ind)) #Fs=X.M.u_s

    contribution_mod1= np.zeros(fact_ind.shape)

    for i in range(fact_ind.shape[1]):
        f=fact_ind[:,i] #f correspond à la colonne
        contribution_mod1[:,i]=100*D.dot(f*f) / f.T.dot(D.dot(f))

    
    distance= (fact_ind **2).sum(1).reshape(fact_ind.shape[0],1) 
    qualite_ind = 100*(fact_ind**2) / distance 

    inerties = 100* val_p_ind / val_p_ind.sum()

    #diagramem des inerties 
    fig1=figure(figsize=(6, 4))
    title('Diagramme des inerties')
    xlabel('Composantes')
    ylabel('Pourcentage d inerties')
    grid(True)
    plot(inerties)
    show()

    #On affiche les individus sur le plan factoriel et les axes factoriels
    fig2=figure(figsize=(6, 4))
    title('Individus sur le plan factoriel')
    xlabel('F1')
    ylabel('F2')
    plot(fact_ind[:,0],fact_ind[:,1],'o')
    #afficher l'axe 0,0
    plot([0,0],[fact_ind[:,1].min(),fact_ind[:,1].max()],'k')
    plot([fact_ind[:,0].min(),fact_ind[:,0].max()],[0,0],'k')
    for i in range(nb_ligne):
        text(fact_ind[i,0],fact_ind[i,1],nomIndividus[i])
    show()



    #On compte le nombre d'axes d'inerties qu'il faut utiliser dans la CAH afin d'obtenir 90% de l'inerties
    somme=0
    compteurAxes=0
    while(somme<90):
        somme+=inerties[compteurAxes]
        compteurAxes+=1

    #On effectue la CAH
    Z = linkage(fact_ind[:,:compteurAxes],method='ward',metric='euclidean')
    #On affiche le dendrogramme
    fig1=figure(figsize=(6, 4))
    title('CAH')
    xlabel('Individus')
    ylabel('Distance')
    dendrogram(
        Z,
        labels=nom_individus,
        leaf_rotation=90.,  # fait pivoter les étiquettes de l'axe des x
        leaf_font_size=8.,  # taille de la police pour les étiquettes de l'axe x
    )
    show()
    #On effectue le clustering
    cluster = fcluster(Z,k,criterion='maxclust')
    print(cluster)
    #On affiche les individus par cluster
    for i in range(1,k+1):
        print("Individus du cluster ",i,":")
        for j in range(len(cluster)):
            if cluster[j]==i:
                print(nom_individus[j])
        print("")
    #Affichage du nuage de points des individus par cluster
    fig2=figure(figsize=(6, 4))
    title('Individus par cluster')
    xlabel('F1')
    ylabel('F2')
    for i in range(1,k+1):
        plot(fact_ind[cluster==i,0],fact_ind[cluster==i,1],'o')
    #afficher l'axe 0,0
    plot([0,0],[fact_ind[:,1].min(),fact_ind[:,1].max()],'k')
    plot([fact_ind[:,0].min(),fact_ind[:,0].max()],[0,0],'k')
    for i in range(nb_ligne):
        text(fact_ind[i,0],fact_ind[i,1],nomIndividus[i])
    show()
    
    print("Fin de la troisième méthode : ACM + CAH")
    print("==================================================")

    return None

    




#%%
#===========================================================================================================================#
#=============================-Cette méthode correspond à un mix de ACM + Centre mobiles-===================================#

def methode_mixte4(data,nom_individus,nom_variables,k):
# Je décide de mettre k en paramètre pour le modifier plus facilement.
# k est le nombre de groupes que fournit la classification non supervisée

    X=normalisation(data)
    #On effectue l'ACM
    print("==================================================")
    print("Lancement de la quatrième méthode : ACM + Centres mobiles")
    [nb_ligne,nb_colone]=np.shape(data)
    X=qttitaf_en_quali_1(X,5)
    nb_mod_par_var = data.max(0)
    nb_mod=int(nb_mod_par_var.sum())

    XTDC = np.zeros((data.shape[0],nb_mod),dtype="int")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            XTDC[i,int(data[i,j]-1 + nb_mod_par_var[:j].sum())]=1

    # print("tableau de burt :")
    # print(mat)
    mat=XTDC

    Xfreq = mat / mat.sum()
    marge_colonne = Xfreq.sum(1).reshape(Xfreq.shape[0],1)
    marge_ligne = Xfreq.sum(0).reshape(1,Xfreq.shape[1])

    Xindep = marge_ligne * marge_colonne
    X = Xfreq/Xindep - 1
    M = np.diag(marge_ligne[0,:])
    D = np.diag(marge_colonne[:,0])


    Xcov_ind=X.T.dot(D.dot(X.dot(M)))

    L,U=np.linalg.eig(Xcov_ind) #récupere les valeurs propres et les vecteurs propres
    L=np.float16(L)
    U=np.float16(U)
    
    indices=np.argsort(L)[: :-1]
    val_p_ind=np.sort(L)[: :-1]
    vect_p_ind=U[:,indices]

    Xcov_var=X.dot(M.dot(X.T.dot(D)))
    L,U=np.linalg.eig(Xcov_var)

    #tri par ordre décroissant des valeurs propres
    val_p_var=np.sort(L)[: :-1]


    fact_ind= X.dot(M.dot(vect_p_ind)) #Fs=X.M.u_s
    

    contribution_mod1= np.zeros(fact_ind.shape)



    for i in range(fact_ind.shape[1]):
        f=fact_ind[:,i] #f correspond à la colonne
        contribution_mod1[:,i]=100*D.dot(f*f) / f.T.dot(D.dot(f))





    
    distance= (fact_ind **2).sum(1).reshape(fact_ind.shape[0],1) 
    qualite_ind = 100*(fact_ind**2) / distance 

    inerties = 100* val_p_ind / val_p_ind.sum()

    #diagramem des inerties 
    fig1=figure(figsize=(6, 4))
    title('Diagramme des inerties')
    xlabel('Composantes')
    ylabel('Pourcentage d inerties')
    plot(inerties)
    show()

    #On affiche les individus sur le plan factoriel et les axes factoriels
    fig2=figure(figsize=(6, 4))
    title('Individus sur le plan factoriel')
    xlabel('F1')
    ylabel('F2')
    plot(fact_ind[:,0],fact_ind[:,1],'o')
    #afficher l'axe 0,0
    plot([0,0],[fact_ind[:,1].min(),fact_ind[:,1].max()],'k')
    plot([fact_ind[:,0].min(),fact_ind[:,0].max()],[0,0],'k')
    for i in range(nb_ligne):
        text(fact_ind[i,0],fact_ind[i,1],nomIndividus[i])
    show()
    
    centre, labels = kmeans2(fact_ind[:,0:i],k)
	
    figure(figsize=(10, 8))
    scatter(fact_ind[:,0], fact_ind[:,1], c=labels)
    for label,x,y in zip(nom_individus,fact_ind[:,0],fact_ind[:,1]):
        annotate(label,
			xy=(x,y),
			xytext=(-50,5),
			textcoords='offset points',
			ha='right', va='bottom',
			arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
			)
        title('Combinaison méthodes ACM + Centres mobiles')


    scatter(centre[:,0],centre[:,1], c = 'r')
    grid(True)
    axvline(linewidth=0.5,color='k')
    axhline(linewidth=0.5,color='k')
    
    show()

    print("Fin de la quatrième méthode : ACM + Centres mobiles")
    print("==================================================")


#%% Tests des méthodes mixtes d'analyse de données et de classification non supervisée

if __name__ == '__main__':
    
    ## ===== Lecture des données ===== ##
    dataNonNormalise=np.loadtxt('donnees/population_donnees.txt')
    
    # On normalise les données pour pouvoir les exploiter dans le cas de l'ACP
    data=normalisation(dataNonNormalise)
    
    # On importe également les noms des variables
    nomIndividus = np.loadtxt('donnees/population_noms_individus.txt',dtype='str')
    nomModalite = np.loadtxt('donnees/population_noms_variables.txt',dtype='str')
    
    
    ## ====== Test des méthodes ACP + (CAH ou Centres mobiles) ====== ##
    
    # k est le nombre de classes recherchées à la fin de la classification
    k=3
    
    premiereMethode(dataNonNormalise,nomIndividus,nomModalite,k)
    methode_mixte2(data,nomIndividus,nomModalite,k)
    
    ## ====== Test des méthodes ACM + (CAH ou Centres mobiles) ====== ##
    
    troisiemeMehode(dataNonNormalise,nomIndividus,nomModalite,k)
    methode_mixte4(data,nomIndividus,nomModalite,k)
