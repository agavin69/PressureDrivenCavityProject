# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 02:07:00 2019

@author: antho
"""

# EGM 7819 Project 2
# Lid Driven Cavity Flow
# Anthony Gavin

import numpy
from matplotlib import pyplot
#from openpyxl import load_workbook


def possion_SOR(p, dx, dy, b):
    pn = numpy.empty_like(p)
    pn = p.copy()
    error = numpy.empty_like(p)
    error_sum = numpy.empty_like(p)
    tol = 1e-5

    psi=1.0-2.0*(numpy.sin(numpy.pi*dx/2.0))**2
    w=2.0/(1.0+numpy.sqrt(1.0-psi))

    for q in range(nit):
        pn = p
        p[1:-1, 1:-1] = ((((pn[1:-1,2:] + pn[1:-1,0:-2])*dy**2 + (pn[2:,1:-1] + pn[0:-2,1:-1])*dx**2) -
                          (dx**2 * dy**2)*b[1:-1,1:-1])/(2 * (dx**2 + dy**2)))
        error[1:-1,1:-1] = p[1:-1,1:-1] - pn[1:-1,1:-1]
        error_sum[1:-1,1:-1] = error[1:-1,1:-1]**2
        p[1:-1, 1:-1] = pn[1:-1,1:-1] + w*error[1:-1,1:-1]

        p[:,-1] = p[:,-2]
        p[0,:] = p[1,:]
        p[:,0] = p[:,1]
        p[-1, :] = p[-2,:]



    return p

def divergence(u,v,dx,dy):
    div = numpy.empty_like(u)

    div[1:-1,1:-1] = (u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx)  +  (v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy)

    div_diag1 = numpy.zeros((numpy.size(u,0),1))
    div_diag2 = numpy.zeros((numpy.size(u,0),1))

    for i in range(div_diag1.size):
        for j in range(div_diag1.size):
            if (i == j):
                div_diag1[i,0] = div[i,j]
                div_diag2[i,0] = div[i,-j]



    return div_diag1,div_diag2



def main(time_iter,u,v,dt,dx,dy,p,Re): # Upwind for Convection
    un = numpy.empty_like(u) # Old u velocity (time step n)
    vn = numpy.empty_like(v) # Old v velocity (time step n)
    b  = numpy.zeros((ny,nx))

    for n in range(time_iter):
        un = u.copy()
        vn = v.copy()

        b[1:-1,1:-1] = ((1/dt * ((u[1:-1,2:] - u[1:-1,0:-2])/(2*dx) + (v[2:,1:-1] - v[0:-2,1:-1])/(2*dy))) -
                   ((u[1:-1,2:] - u[1:-1,0:-2])/(2*dx))**2 - 2*((u[2:,1:-1] - u[0:-2,1:-1])/(2*dy) *
                   (v[1:-1, 2:] - v[1:-1,0:-2])/(2*dx))-((v[2:,1:-1] - v[0:-2,1:-1])/(2*dy))**2)

        p = possion_SOR(p, dx, dy, b)

        u[1:-1,1:-1] = (un[1:-1,1:-1] - un[1:-1,1:-1]*(dt/dx)*(un[1:-1,1:-1] - un[1:-1,0:-2]) - # Previous velocity and convective terms
                       vn[1:-1,1:-1]*(dt/dy)*(un[1:-1,1:-1] - un[0:-2,1:-1]) -
                       dt/(2*dx) * (p[1:-1,2:] - p[1:-1,0:-2]) + # Pressure terms
                       (1/Re) * ((dt/dx**2) *(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]) + # Diffusive terms
                       (dt/dy**2) *(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1])))

        v[1:-1,1:-1] = (vn[1:-1,1:-1] - un[1:-1,1:-1]*(dt/dx)* # Previous velocity and convective terms
                       (vn[1:-1,1:-1] - vn[1:-1,0:-2]) - vn[1:-1,1:-1]*(dt/dy)*
                       (vn[1:-1,1:-1] - vn[0:-2,1:-1]) -
                       dt/(2*dy) * (p[2:,1:-1] - p[0:-2,1:-1]) +  # Pressure term
                       (1/Re) * ((dt/dx**2) *(vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1, 0:-2]) +  # Diffusive terms
                       (dt/dy**2) * (vn[2:,1:-1] - 2 * vn[1:-1,1:-1] + vn[0:-2,1:-1])))


        # Update Boundary Conditions
        u[0,:] = 0
        u[:,0] = 0
        u[:,-1] = 0
        u[-1,:] = 1
        v[0,:] = 0
        v[-1,:]=0
        v[:,0] = 0
        v[:,-1] = 0


    return u,v,p

def main_Central(time_iter,u,v,dt,dx,dy,p,Re): # Central Differencing for convection
    un = numpy.empty_like(u) # Old u velocity (time step n)
    vn = numpy.empty_like(v) # Old v velocity (time step n)
    b  = numpy.zeros((ny,nx))

    for n in range(time_iter):
        un = u.copy()
        vn = v.copy()

        b[1:-1,1:-1] = ((1/dt * ((u[1:-1,2:] - u[1:-1,0:-2])/(2*dx) + (v[2:,1:-1] - v[0:-2,1:-1])/(2*dy))) -
                   ((u[1:-1,2:] - u[1:-1,0:-2])/(2*dx))**2 - 2*((u[2:,1:-1] - u[0:-2,1:-1])/(2*dy) *
                   (v[1:-1, 2:] - v[1:-1,0:-2])/(2*dx))-((v[2:,1:-1] - v[0:-2,1:-1])/(2*dy))**2)

        p = possion_SOR(p, dx, dy, b)

        u[1:-1,1:-1] = (un[1:-1,1:-1] - un[1:-1,1:-1]*(dt/(dx))*(un[1:-1,2:] - un[1:-1,0:-2]) - # Previous velocity and convective terms
                       vn[1:-1,1:-1]*(dt/(dy))*(un[2:,1:-1] - un[0:-2,1:-1]) -
                       dt/(2*dx) * (p[1:-1,2:] - p[1:-1,0:-2]) + # Pressure terms
                       (1/Re) * ((dt/dx**2) *(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]) + # Diffusive terms
                       (dt/dy**2) *(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1])))

        v[1:-1,1:-1] = (vn[1:-1,1:-1] - un[1:-1,1:-1]*(dt/(dx))* # Previous velocity and convective terms
                       (vn[1:-1,2:] - vn[1:-1,0:-2]) - vn[1:-1,1:-1]*(dt/(dy))*
                       (vn[2:,1:-1] - vn[0:-2,1:-1]) -
                       dt/(2*dy) * (p[2:,1:-1] - p[0:-2,1:-1]) +  # Pressure term
                       (1/Re) * ((dt/dx**2) *(vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1, 0:-2]) +  # Diffusive terms
                       (dt/dy**2) * (vn[2:,1:-1] - 2 * vn[1:-1,1:-1] + vn[0:-2,1:-1])))


        # Update Boundary Conditions
        u[0,:] = 0
        u[:,0] = 0
        u[:,-1] = 0
        u[-1,:] = 1
        v[0,:] = 0
        v[-1,:]=0
        v[:,0] = 0
        v[:,-1] = 0


    return u,v,p

def visual(u,v,p,X,Y,u51,v51,p51,u200,v200,p200,x51,y51,x200,y200,div_diag1_51,div_diag2_51,div_diag1_100,div_diag2_100,div_diag1_200,div_diag2_200,u200_f,v200_f,p200_f,x200_f,y200_f,dx,dy):

    pyplot.figure(1)
    pyplot.contourf(X,Y,u,100)
    pyplot.title("u Velocity")
    pyplot.ylabel("Y")
    pyplot.xlabel("X")
    pyplot.colorbar()
    pyplot.show()

    pyplot.figure(2)
    pyplot.contourf(X,Y,v,100)
    pyplot.title("v Velocity")
    pyplot.ylabel("Y")
    pyplot.xlabel("X")
    pyplot.colorbar()
    pyplot.show()

    pyplot.figure(3)
    pyplot.contourf(X,Y,p,100)
    pyplot.title("Pressure")
    pyplot.ylabel("Y")
    pyplot.xlabel("X")
    pyplot.colorbar()
    pyplot.show()

    #path = 'C:\Users\antho\OneDrive\Documents\Adv CFD' # Home Computer


    # if file is saved in the same directory simply use: path = 'data.xlsx'
    #wb   = load_workbook(path2,guess_types = True)
    #ws   = wb.active
    #ghia_u = numpy.zeros((15-2,1))
    #ghia_y = numpy.zeros((15-2,1))
    #ghia_x = numpy.zeros((22-2,1))
    #ghia_v = numpy.zeros((22-2,1))

    #for i in range(ghia_u.size):
        #cell1 = 'A' + str(i+3)
        #cell2 = 'B' + str(i+3)
        #ghia_u[i][0] = ws[cell1].value
        #ghia_y[i][0] = ws[cell2].value

    #for i in range(ghia_v.size):
      #  cell1 = 'C' + str(i+3)
       # cell2 = 'D' + str(i+3)
        #ghia_x[i][0] = ws[cell1].value
        #ghia_v[i][0] = ws[cell2].value

    pyplot.figure(4)
    #pyplot.plot(u51[:,25],y51,'b',u[:,51],y,'r',u200[:,101],y200,'g',ghia_u,ghia_y,'k x')
    pyplot.title("u Velocity Along  x = 0.5 Centerline")
    pyplot.legend(["N = 51","N = 100","N = 200","Ghia et. al."])
    pyplot.ylabel("y")
    pyplot.xlabel("u")
    pyplot.show()

    pyplot.figure(5)
    #pyplot.plot(x51,v51[25,:],'b',x,v[51,:],'r',x200,v200[101,:],'g',ghia_x,ghia_v,'k x')
    pyplot.title("v Velocity Along  y = 0.5 Centerline")
    pyplot.legend(["N = 51","N = 100","N = 200","Ghia et. al."])
    pyplot.ylabel("v")
    pyplot.xlabel("x")
    pyplot.show()

    # Streamlines 
    pyplot.figure(6)
    pyplot.streamplot(X[::2,::2],Y[::2,::2],u[::2,::2],v[::2,::2],3)
    pyplot.title("Streamlines")
    pyplot.ylabel("Y")
    pyplot.xlabel("X")
    pyplot.show()

    # Vector Plot of Velocity field
    pyplot.figure(7)
    pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    pyplot.title("Vector Plot of Velocity Field")
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    pyplot.show()

    # Divergence on Diagonals
    pyplot.figure(8)
    pyplot.plot(x51,div_diag1_51,'k',x51,div_diag2_51,'k o',x,div_diag1_100,'b',x,div_diag2_100,'b x',x200,div_diag1_200,'r',x200,div_diag2_200,'r *')
    pyplot.title("Divergence on Diagonals")
    pyplot.xlabel('Diagonal (0 = bottom corner, 1 = top corner)')
    pyplot.ylabel(r"$\nabla \cdot u$")
    pyplot.legend(["N = 51 Diagonal 1","N = 51 Diagonal 2","N = 100 Diagonal 1","N = 100 Diagonal 2","N = 200 Diagonal 1","N = 200 Diagonal 2"])
    pyplot.show()

    # Peclet number \rho u dx/ \mu

    mu = 1.81e-5 #assuming air?
    rho = 1.25 # assuming air?

    pyplot.figure(9)
    pyplot.contourf(x200,y200,numpy.abs(((rho*(numpy.sqrt(u200**2 + v200**2))*dx)/mu)-((rho*(numpy.sqrt(u200_f**2 + v200_f**2))*dx)/mu)),100)
    pyplot.title("Absolute value of the Difference in Cell Peclet Numbers")
    pyplot.ylabel("Y")
    pyplot.xlabel("X")
    pyplot.colorbar()
    pyplot.show()

    return


Re = 100
nx = 101
ny = 101
time_iter = 15000
nit = 400
dx = 1/ (nx - 1)
dy = 1 / (ny - 1)
x = numpy.linspace(0, 1, nx)
y = numpy.linspace(0, 1, ny)
X, Y = numpy.meshgrid(x, y)


dt = .0001

u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx))
b = numpy.zeros((ny, nx))
u,v,p = main_Central(time_iter, u, v, dt, dx, dy, p, Re)
div_diag1_100,div_diag2_100 = divergence(u,v,dx,dy)

nx = 51
ny = 51
dx = 1/ (nx - 1)
dy = 1 / (ny - 1)
u51 = numpy.zeros((ny, nx))
v51 = numpy.zeros((ny, nx))
p51 = numpy.zeros((ny, nx))
x51 = numpy.linspace(0, 1, nx)
y51 = numpy.linspace(0, 1, ny)
u51,v51,p51 = main_Central(time_iter, u51, v51, dt, dx, dy, p51, Re)
div_diag1_51,div_diag2_51 = divergence(u51,v51,dx,dy)

nx = 200
ny = 200
dx = 1/ (nx - 1)
dy = 1 / (ny - 1)
u200 = numpy.zeros((ny, nx))
v200 = numpy.zeros((ny, nx))
p200 = numpy.zeros((ny, nx))
x200 = numpy.linspace(0, 1, nx)
y200 = numpy.linspace(0, 1, ny)
u200,v200,p200 = main_Central(time_iter, u200, v200, dt, dx, dy, p200, Re)
div_diag1_200,div_diag2_200 = divergence(u200,v200,dx,dy)

u200_f = numpy.zeros((ny, nx))
v200_f = numpy.zeros((ny, nx))
p200_f = numpy.zeros((ny, nx))
x200_f = numpy.linspace(0, 1, nx)
y200_f = numpy.linspace(0, 1, ny)
u200_f,v200_f,p200_f = main(time_iter, u200_f, v200_f, dt, dx, dy, p200_f, Re)


visual(u,v,p,X,Y,u51,v51,p51,u200,v200,p200,x51,y51,x200,y200,div_diag1_51,div_diag2_51,div_diag1_100,div_diag2_100,div_diag1_200,div_diag2_200,u200_f,v200_f,p200_f,x200_f,y200_f,dx,dy)

print('end')
