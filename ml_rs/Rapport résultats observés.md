# **Rapport sur les résultats observés**

## **Sommaire**

1. [Modèle linéaire](#modèle-linéaire)
    - [Classification](#classification-linéaire)
    - [Régression](#régression-linéaire)

2. [Perceptron Multi-Couches](#perceptron-multi-couches)
    - [Classification](#classification-mlp)
    - [Régression](#régression-mlp)

---

# Modèle Linéaire

## Classification <a name="classification-linéaire"></a>

### 1. Linear Simple

Le problème est linéairement séparable.  
**Résultat :** le modèle linéaire prédit correctement les sorties, convergence rapide.

### 2. Linear Multiple

Deux clusters clairement séparables par une frontière linéaire.  
**Résultat :** le perceptron linéaire réussit la classification malgré le bruit.

### 3. XOR

Problème **non linéaire**, impossible à séparer par une droite unique.  
**Résultat :** échec attendu, prédictions incohérentes.

### 4. Cross

Frontière en forme de croix, donc structure fortement non linéaire.  
**Résultat :** le modèle linéaire ne parvient pas à capturer la forme ; performances faibles.

### 5. Three Classes (One-vs-All)

Trois classes séparées selon des règles géométriques non triviales.  
Grâce à trois perceptrons linéaires indépendants (one-vs-all), chacun apprend à reconnaître *sa* classe.  
**Résultat :** le système parvient étonnamment bien à classer correctement, malgré la non-linéarité du problème.  
Il reste des erreurs, mais le taux de réussite est élevé.

### 6. Multi Cross

Structure périodique complexe, dépendant du modulo.  
**Résultat :** les frontières sont trop non linéaires pour un modèle linéaire ; classification incorrecte dans la majorité des cas.

---

## Régression <a name="régression-linéaire"></a>

### 1. Linear Simple

Relation parfaitement linéaire.  
**Résultat :** très bon fit, erreur quasiment nulle.

### 2. Non-Linear Simple

Courbe non linéaire.  
**Résultat :** la régression linéaire ne peut approximer que par une droite → erreur systématique.

### 3. Linear Simple 3D

Relation linéaire entre deux entrées et une sortie.  
**Résultat :** bonne approximation, convergence correcte.

### 4. Linear Tricky 3D

Toujours linéaire, mais dans une configuration moins directe.  
**Résultat :** le modèle converge bien, résultats conformes.

### 5. Non-Linear Simple 3D

Structure non linéaire entre deux entrées et une sortie.  
**Résultat :** le modèle linéaire échoue logiquement → fit approximatif.

---

# Perceptron Multi-Couches (MLP)

## Classification <a name="classification-mlp"></a>

### 1. Linear Simple

Le MLP reproduit parfaitement les résultats du modèle linéaire.  
**Résultat :** séparation triviale, haute précision.

### 2. Linear Multiple

Structure simple.  
**Résultat :** très bonne convergence, prédictions correctes.

### 3. XOR

Cas typique démontrant le pouvoir du MLP.  
**Résultat :** le réseau apprend facilement la relation XOR.

### 4. Cross

Structure non linéaire.  
**Résultat :** un petit MLP parvient à apprendre la forme avec une bonne précision.

### 5. Three Classes

Frontières non triviales.  
**Résultat :** le MLP capture bien les trois classes avec peu d’erreur.

### 6. Multi Cross

Structure périodique complexe (modulo).  
**Résultat :**  
- (2, 16, 16, 3) ≈ ~80–85% correct  
- (2, 32, 32, 3) + entraînement long → ~95% correct  
Le modèle finit par reproduire la structure périodique.

---

## Régression <a name="régression-mlp"></a>

### 1. Linear Simple

Le MLP reproduit facilement une relation linéaire.  
**Résultat :** excellent fit.

### 2. Non-Linear Simple

Le test nécessite une courbure.  
**Résultat :** un petit MLP (1–3–1) capture parfaitement les non-linéarités.

### 3. Linear Simple 3D

Relation linéaire dans ℝ².  
**Résultat :** apprentissage immédiat.

### 4. Linear Tricky 3D

Toujours linéaire.  
**Résultat :** convergence facile.

### 5. Non-Linear Simple 3D

Relation non linéaire.  
**Résultat :** le MLP apprend la structure sans difficulté, contrairement au modèle linéaire.

---

# **Conclusion**

- Le modèle linéaire fonctionne très bien sur les structures linéaires,  
  et étonnamment bien en multiclasse via one-vs-all pour le test 5.

- Le MLP surpasse largement le modèle linéaire sur tout ce qui est non linéaire  
  (XOR, cross, patterns périodiques, régression non linéaire).

- Le MLP devient quasi universel dès qu’on augmente légèrement la capacité  
  (ex : architecture (2, 32, 32, 3)).

