''' Ruler 1         2         3         4         5         6         7        '
/*******************************************************************************
*                                                                              *
*               ██╗   ██╗ ██╗ ███████╗ ██╗  ██████╗  ███╗   ██╗                *
*               ██║   ██║ ██║ ██╔════╝ ██║ ██╔═══██╗ ████╗  ██║                *
*               ██║   ██║ ██║ ███████╗ ██║ ██║   ██║ ██╔██╗ ██║                *
*               ╚██╗ ██╔╝ ██║ ╚════██║ ██║ ██║   ██║ ██║╚██╗██║                *
*                ╚████╔╝  ██║ ███████║ ██║ ╚██████╔╝ ██║ ╚████║                *
*                 ╚═══╝   ╚═╝ ╚══════╝ ╚═╝  ╚═════╝  ╚═╝  ╚═══╝                *
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

from flag import *


#------------------------------------------------------------------------------#
#                                       MAIN                                   #
#------------------------------------------------------------------------------#

if __name__ == "__main__":

    flag_filename = 'flag1.png'
    flag = Bandera(flag_filename)
    # number_of_colors = flag.Colores(graph=True)     # plot inertial graphic
    # percentage = flag.Porcentaje(verbose=True)       # print percentage
    # orientation_flag = flag.Orientacion(graph=True) # plot lines found
