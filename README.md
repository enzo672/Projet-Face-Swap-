# Projet-Face-Swap-

Prototype de transfert d’identité faciale basé sur un encodeur partagé et deux décodeurs. Développé avec PyTorch et OpenCV pour la recherche sur l’alignement, la reconstruction et la détection éthique de deepfakes.

Explication des différents fichiers : 

étape 1 : face_extraction.py -> Extraction frame par frame, détection et alignement des visages <br>
étape 2 : face_extraction.py -> Alignement + masque (face_masking.py)<br>
étape 3 : entrainement du model -> q96 <br>
étape 4 : Créer la vidéo fake -> merge_frames_to_fake_video


