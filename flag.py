''' Ruler 1         2         3         4         5         6         7        '
/*******************************************************************************
*                                                                              *
*       ██████╗   █████╗  ███╗   ██╗ ██████╗  ███████╗ ██████╗   █████╗        *
*       ██╔══██╗ ██╔══██╗ ████╗  ██║ ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗       *
*       ██████╔╝ ███████║ ██╔██╗ ██║ ██║  ██║ █████╗   ██████╔╝ ███████║       *
*       ██╔══██╗ ██╔══██║ ██║╚██╗██║ ██║  ██║ ██╔══╝   ██╔══██╗ ██╔══██║       *
*       ██████╔╝ ██║  ██║ ██║ ╚████║ ██████╔╝ ███████╗ ██║  ██║ ██║  ██║       *
*       ╚═════╝  ╚═╝  ╚═╝ ╚═╝  ╚═══╝ ╚═════╝  ╚══════╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝       *
*                                                                              *
*                  Developed by:                                               *
*                                                                              *
*                            Jhon Hader Fernandez                              *
*                     - jhon_fernandez@javeriana.edu.co                        *
*                                                                              *
*                       Pontificia Universidad Javeriana                       *
*                            Bogota DC - Colombia                              *
*                                  Nov - 2020                                  *
*                                                                              *
*****************************************************************************'''

#------------------------------------------------------------------------------#
#                          IMPORT MODULES AND LIBRARIES                        #
#------------------------------------------------------------------------------#

from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from time import time
import numpy as np
import os
import sys
import cv2


#------------------------------------------------------------------------------#
#                                 Bandera Class                                #
#------------------------------------------------------------------------------#

class Bandera:

    def __init__(self, flag_image):
        self.flag = cv2.imread(flag_image)
        self.width = self.flag.shape[0]
        self.height = self.flag.shape[1]
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.theta = np.arange(0, 360, 0.5)


    # +======================================================================+ #
    # |                                  SHOW                                | #
    # +======================================================================+ #
    def show(self):
        cv2.imshow('Flag', self.flag)
        cv2.waitKey(0)


    # +======================================================================+ #
    # |                                COLORES                               | #
    # +======================================================================+ #
    def Colores(self, graph=True):
        max_colors = 4
        distances = []
        for i in range(1, max_colors+1):
            [labels, centers, inertia] = self.__color_segmentation(method='kmeans', n_colors=i, recreate=False)
            distances.append(inertia)
        number_of_colors = distances.index(min(distances)) + 1

        if graph:
            # se grafica la suma de distancias intra-cluster vs numero de clusters/gaussianas
            plt.plot(np.linspace(start=1, stop=max_colors, num=max_colors), np.array(distances))
            plt.title('Inertia vs Number of colors')
            plt.xlabel('Number of colors')
            plt.ylabel('Inertia')
            plt.show()

            print('\n+------------------------+')
            print('| Number of colors is:', number_of_colors, '|')
            print('+------------------------+\n')

        return number_of_colors


    # +======================================================================+ #
    # |                          COLOR SEGMENTATION                          | #
    # +======================================================================+ #
    def __color_segmentation(self, method='kmeans', n_colors=2, recreate=False):
        # Convert image BGR to RGB
        image = cv2.cvtColor(self.flag.copy(), cv2.COLOR_BGR2RGB)

        # Verify K (number of clusters) it has to be greater than 0
        if n_colors == 0:
            print('[ERROR...] "n_colors" has to be grater than 0')
            sys.exit()

        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        image = np.array(image, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))

        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:10000]
        if method == 'gmm':
            print("[INFO...] Fitting Kmeans model")
            model = GMM(n_components=n_colors).fit(image_array_sample)
            print("[INFO...] Predicting color indices on the full image (GMM)")
        elif method == 'kmeans':
            print("[INFO...] Fitting Kmeans model")
            model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
            print("[INFO...] Predicting color indices on the full image (K means)")
        else:
            print('[ERROR...] Invalid method: select "gmm" or "kmeans"')
            sys.exit()
        print("[REPORT...] done in %0.3fs." % (time() - t0))

        # Get labels for all points
        t0 = time()
        if method == 'gmm':
            labels = model.predict(image_array)
            centers = model.means_
            inertia = model.inertia_
        else:
            labels = model.predict(image_array)
            centers = model.cluster_centers_
            inertia = model.inertia_
        print('[REPORT...] results was obteined')
        print("[REPORT...] done in %0.3fs." % (time() - t0))


        if recreate:
            # Display all results, alongside original image
            plt.figure(1)
            plt.clf()
            plt.axis('off')
            plt.title('Original image')
            plt.imshow(image)

            plt.figure(2)
            plt.clf()
            plt.axis('off')
            plt.title('Quantized image ({} colors, method={})'.format(n_colors, method))
            plt.imshow(self.__recreate_image(centers, labels, rows, cols))

            plt.show()
        return [labels, centers, inertia]


    def __recreate_image(self, centers, labels, rows, cols):
        d = centers.shape[1]
        image_clusters = np.zeros((rows, cols, d))
        label_idx = 0
        for i in range(rows):
            for j in range(cols):
                image_clusters[i][j] = centers[labels[label_idx]]
                label_idx += 1

        return image_clusters


    # +======================================================================+ #
    # |                              PORCENTAJE                              | #
    # +======================================================================+ #
    def Porcentaje(self, verbose=True):
        max_colors = 4
        distances = []
        labels_list = []
        centers_list = []
        for i in range(1, max_colors+1):
            [labels, centers, inertia] = self.__color_segmentation(method='kmeans', n_colors=i, recreate=False)
            labels_list.append(labels)
            distances.append(inertia)
        number_of_colors = distances.index(min(distances)) + 1

        labels = labels_list[number_of_colors - 1]
        percentage = []

        for i in range(0, max_colors):
            count = list(labels).count(i)
            percentage.append(count)

        percentage = np.array(percentage) / (self.flag.shape[0] * self.flag.shape[1])
        percentage = np.round(percentage * 100.0, 3)

        if verbose:
            print('\n Percentage: ')
            print(percentage, '\n')

        return percentage


    # +======================================================================+ #
    # |                             ORIENTACION                              | #
    # +======================================================================+ #
    def Orientacion(self, graph=False):

        # Set 1 degree to tolerance
        tolerance = 1 # degree

                       # ╔═════════════╦═════════════╗ #
                       # ║ Theta (deg) ║ Orientation ║ #
                       # ╠═════════════╬═════════════╣ #
                       # ║  180 ± tol  ║   vertical  ║ #
                       # ╠═════════════╬═════════════╣ #
                       # ║   90 ± tol  ║  horizontal ║ #
                       # ╠═════════════╬═════════════╣ #
                       # ║  any ohter  ║   mixture   ║ #
                       # ╚═════════════╩═════════════╝ #

        theta = self.__hough(N_peaks=3, graph=graph)
        if (theta < 90.0 + tolerance) and (theta > 90.0 - tolerance):
            orientation = 'horizontal'
        elif (theta < 180.0 + tolerance) and (theta > 180.0- tolerance):
            orientation = 'vertical'
        else:
            orientation = 'mixture'

        print('\nOrientation is:', orientation, '\n')

        return orientation


    # +======================================================================+ #
    # |                                 HOUGH                                | #
    # +======================================================================+ #
    def __hough(self, N_peaks=1, graph=True):
        # Get edges
        high_thresh = 300
        bw_edges = cv2.Canny(self.flag, high_thresh * 0.3, high_thresh, L2gradient=True)

        # Get accumulator
        accumulator = self.__standard_HT(bw_edges)

        # Get peaks
        acc_thresh = 50
        # N_peaks = 0   # <- Code
        nhood = [25, 9]
        peaks = self.__find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        # Draw lines found
        [_, cols] = self.flag.shape[:2]
        image_draw = np.copy(self.flag)
        for i in range(len(peaks)):
            rho = peaks[i][0]
            theta_ = self.theta[peaks[i][1]]

            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + self.center_x
            y0 = b * rho + self.center_y
            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))

            if np.abs(theta_) < 80:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
            elif np.abs(theta_) > 100:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
            else:
                if theta_ > 0:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 0], thickness=2)
                else:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)


        if graph:
            cv2.imshow("Edges", bw_edges)
            cv2.imshow("Lines over frame", image_draw)
            cv2.waitKey(0)

        return theta_


    def __standard_HT(self, bw_edges):
        rows = bw_edges.shape[0]
        cols = bw_edges.shape[1]
        rmax = int(round(0.5 * np.sqrt(rows ** 2 + cols ** 2)))
        y, x = np.where(bw_edges >= 1)

        accumulator = np.zeros((rmax, len(self.theta)))

        for idx, th in enumerate(self.theta):
            r = np.around(
                (x - self.center_x) * np.cos((th * np.pi) / 180) + (y - self.center_y) * np.sin((th * np.pi) / 180))
            r = r.astype(int)
            r_idx = np.where(np.logical_and(r >= 0, r < rmax))
            np.add.at(accumulator[:, idx], r[r_idx[0]], 1)

        return accumulator


    def __find_peaks(self, accumulator, nhood, accumulator_threshold, N_peaks):
        done = False
        acc_copy = accumulator
        nhood_center = [(nhood[0] - 1) / 2, (nhood[1] - 1) / 2]
        peaks = []
        while not done:
            [p, q] = np.unravel_index(acc_copy.argmax(), acc_copy.shape)
            if acc_copy[p, q] >= accumulator_threshold:
                peaks.append([p, q])
                p1 = p - nhood_center[0]
                p2 = p + nhood_center[0]
                q1 = q - nhood_center[1]
                q2 = q + nhood_center[1]

                [qq, pp] = np.meshgrid(np.arange(np.max([q1, 0]), np.min([q2, acc_copy.shape[1] - 1]) + 1, 1), \
                                       np.arange(np.max([p1, 0]), np.min([p2, acc_copy.shape[0] - 1]) + 1, 1))
                pp = np.array(pp.flatten(), dtype=np.intp)
                qq = np.array(qq.flatten(), dtype=np.intp)

                acc_copy[pp, qq] = 0
                done = np.array(peaks).shape[0] == N_peaks
            else:
                done = True

        return peaks