import sys
import cv2
import dlib 
import matplotlib.pyplot as plt
import os

def adaugaCaciula(pozaOm, pozaCaciula):
    pozaCaciulaGri = cv2.cvtColor(pozaCaciula, cv2.COLOR_BGR2GRAY)
    
    _ret, masca = cv2.threshold(pozaCaciulaGri, 1, 255, cv2.THRESH_BINARY)
    hPozaCaciula, wPozaCaciula, _c = pozaCaciula.shape
    
    pozaCaciula = cv2.cvtColor(pozaCaciula, cv2.COLOR_BGR2RGB)
    pozaOm = cv2.cvtColor(pozaOm, cv2.COLOR_BGR2RGB)
    pozaOmInit = pozaOm.copy()
    
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    # dlib detectia fetei
    detector = dlib.get_frontal_face_detector()
    detectari = detector(pozaOm, 1)
    # daca a gasit macar o figura
    if len(detectari) > 0:
        for d in detectari:
            yFata = d.top()
            wFata = d.right() - d.left()
            forme = predictor(pozaOm, d)
            # edge points of eyes
            ochiDreapta = forme.part(0)
            ochiStanga = forme.part(2)
            # cautam centrul fetei
            centru = ((ochiDreapta.x + ochiStanga.x)//2, (ochiDreapta.y + ochiStanga.y)//2)

            wCaciulaNou = int(wFata * 2.0)
            hCaciulaNou = yFata - 220
            pozaCaciulaNoua = cv2.resize(pozaCaciula, (wCaciulaNou, hCaciulaNou))

            mask = cv2.resize(masca, (wCaciulaNou, hCaciulaNou))
            mask_inv = cv2.bitwise_not(mask)
            mask_inv_triplu = cv2.merge((mask_inv, mask_inv, mask_inv))

            # cat in plus
            hDif = 70
            #extragem ROI ca sa combinam cu pozele in el
            y1ROI = yFata + hDif - hCaciulaNou
            y2ROI = yFata + hDif
            x1ROI = (centru[0] - wCaciulaNou // 3)
            x2ROI = (centru[0] + wCaciulaNou // 3 * 2)
            bgROI = pozaOm[y1ROI : y2ROI, x1ROI:x2ROI]
            bgROICopie = bgROI.copy()

            pozaCaciulaNouaRedimBinar = cv2.resize(mask_inv_triplu, (bgROI.shape[1], bgROI.shape[0]))

            bgMaskROI = cv2.bitwise_and(pozaCaciulaNouaRedimBinar, bgROI)
            pozaCaciulaNoua = cv2.resize(pozaCaciulaNoua, (bgROI.shape[1],bgROI.shape[0]))

            # combina masca cu bg ROI + poza caciula ROI
            caciulaAdaugata = cv2.add(bgMaskROI, pozaCaciulaNoua)
            # cv2.imshow("add_hat",add_hat)
            # aducem inapoi ROI-ul in poza initiala
            pozaOm[y1ROI : y2ROI, x1ROI:x2ROI] = caciulaAdaugata

            fig = plt.figure(figsize=(10, 12))
            ax1 = fig.add_subplot(3,3,1) #numarul de linii, de coloane, index.
            ax1.imshow(pozaOmInit)
            ax1.axis('off')
            ax1.set_title('Initiala')

            8

            ax2 = fig.add_subplot(3, 3, 2)
            ax2.imshow(pozaCaciulaNoua)
            ax2.set_title('Poza caciula redimensionata')
            ax2.axis('off')
            ax3 = fig.add_subplot(3, 3, 3)
            ax3.set_title('masca')
            ax3.imshow(mask, cmap = 'gray')
            ax3.axis('off')
            ax4 = fig.add_subplot(3, 3, 4)
            ax4.set_title('Masca inversa')
            ax4.imshow(mask_inv, cmap = 'gray')
            ax4.axis('off')
            ax5 = fig.add_subplot(3, 3, 5)
            ax5.set_title('ROI')
            ax5.imshow(bgROICopie)
            ax5.axis('off')
            ax6 = fig.add_subplot(3, 3, 6)
            ax6.set_title('Masca inversa redimensionata')
            ax6.imshow(pozaCaciulaNouaRedimBinar, cmap = 'gray')
            ax6.axis('off')
            ax7 = fig.add_subplot(3, 3, 7)
            ax7.set_title('Masca ROI')
            ax7.imshow(bgMaskROI)
            ax7.axis('off')
            ax8 = fig.add_subplot(3, 3, 8)
            ax8.set_title('Caciula adaugata in ROI')
            ax8.imshow(caciulaAdaugata)
            ax8.axis('off')
            ax9 = fig.add_subplot(3, 3, 9)
            ax9.set_title('Sarbatori Fericite!')
            ax9.imshow(pozaOm)
            ax9.axis('off')
        return pozaOm


if os.path.exists('rezultat.jpg'):
    os.remove('rezultat.jpg')

pozaCaciula = cv2.imread('caciula.jpg')
pozaOm = cv2.imread('poza.jpg')
cuCaciula = adaugaCaciula(pozaOm, pozaCaciula)
# putem salva fisierul
pozaOmcuCaciula = cv2.cvtColor(cuCaciula, cv2.COLOR_RGB2BGR)

cv2.imwrite("rezultat.jpg", pozaOmcuCaciula)