import numpy as np
import matplotlib.pyplot as plt
import plot_set
import scipy.integrate as Integrate 



N = 1000

#Masses of particles in Gev
m_m = 0.1057 
m_b = 4.85
m_z = 91.1876 

#Charge
e = 0.31345
Q = -1/3

#Energy in Gev
E_cm = 10
E = E_cm/2
s = E_cm**2

#Momentum
p = np.sqrt(E**2-m_m**2)
pp = np.sqrt(E**2-m_b**2)

#Constants
sin2_tw = 0.231
sin_tw = np.sqrt(sin2_tw)
tw = np.arcsin(sin_tw)
g_z = e/(sin_tw*np.cos(tw))
WidthZ = 2.43631

c_g = Q**2*8*e**4/(s**2)
c_z = 8*g_z**4*1/(s-m_z**2)**2
c_gz =  -Q*8*e**2*g_z**2 *1/((s-m_z**2)*s)

#Axial coupling constants
g_ap = -0.04*0.5
g_bp = -0.5*0.5

g_a = -0.35*0.5
g_b = -0.5*0.5

g_tot = g_ap*g_bp*g_a*g_b 

Omega =  (g_ap**2 + g_bp**2)
Omega_t =  (g_ap**2 - g_bp**2)

Omega_p =  (g_a**2 + g_b**2)
Omega_tp =  (g_a**2 - g_b**2)


#cos_theta = np.cos(np.arccos(np.linspace(-1,1,N)))
cos_theta = np.linspace(-1,1,N)

def Xi():
    return 2*(E**4 + (p*pp)**2 * cos_theta**2) + m_m**2*(pp**2+E**2) + m_b**2*(E**2+ p**2) + 2*m_m**2*m_b**2

def M_g():
    return c_g*Xi() 

def M_z():
    return c_z*(2*Omega*Omega_p*(E**4 + (p*pp)**2*cos_theta**2) + Omega_t*Omega_p*m_m**2*(E**2 + pp**2) \
            + Omega*Omega_tp*m_b**2*(E**2 + p**2) + 2* Omega_t*Omega_p*m_m**2*m_b**2 + 16*g_tot*E**2*p*pp*cos_theta) 

def M_gz():
    return 2*c_gz*(g_ap*g_a*Xi() + g_bp*g_b*4*p*pp*cos_theta*E**2)


def calcCrossSection(*args):
    M_element = np.zeros(N)
    for arg in args:
        M_element += arg()
    return 3/(32*np.pi*s)*1/(2.56810e-9) * pp/p * M_element


plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')

data = np.loadtxt("data/10Gev.txt", skiprows=3)
compHepCos = data[:, 0]
compHepCross = data[:, 1]

plt.plot(cos_theta, calcCrossSection(M_g,M_z,M_gz), label = r"$M_{tot}$")
#plt.plot(cos_theta, calcCrossSection(M_g), label = r"$M_{\gamma}$")
#plt.plot(cos_theta, calcCrossSection(M_z), label = r"$M_{z}$")
#plt.plot(cos_theta, calcCrossSection(M_gz), label = r"$2M_{\gamma,z}$")
plt.plot(compHepCos,compHepCross, "--", label = "Comp Hep")

plt.legend(fontsize = 15)
plt.xlabel(r"$\cos(\theta)$", fontsize=15)
plt.ylabel(r"$d\sigma/dcos\theta$  [pb/rad]", fontsize=15)
plt.title("Differential Cross section with"r"$\sqrt{s}$"f"= {E_cm} for " + r"$\mu^-,\mu^+ \rightarrow b,\bar{b}$", fontsize = 15)
plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.savefig("articles/10_comp.pdf")
plt.show()


M = 200
#Energy in Gev
E_cm_Arr = np.linspace(10,200,M)
"""
Cross section as a function of energy
"""

def totCross(*args):
    global E_cm, E, s, p, pp, c_g, c_z, c_gz
    crossTot = np.zeros(M)
    for i in range(M):
        #Update constants
        E_cm = E_cm_Arr[i]
        E = E_cm/2
        s = E_cm**2

        c_g = Q**2*8*e**4/(s**2)
        c_z = 8*g_z**4*np.real( 1/( (s-m_z**2-1j*m_z*WidthZ)*(s-m_z**2+1j*m_z*WidthZ) ) )#1/(s-m_z**2)**2
        c_gz =  -Q*8*e**2*g_z**2 * np.real(1/(3*s*(s-m_z**2+1j*m_z*WidthZ)))#1/((s-m_z**2)*s)

        p = np.sqrt(E**2-m_m**2)
        pp = np.sqrt(E**2-m_b**2)

        if len(args)>1:
            cross = calcCrossSection(args[0], args[1], args[2])
        else:
            cross = calcCrossSection(args[0])

        crossTot[i] = np.trapz(cross,np.linspace(-1,1,N))
    return crossTot

data = np.loadtxt("data/totCrossCH.txt", skiprows=3)
compHepE = data[:, 0]
compHepCS = data[:, 1]
cross = totCross(M_g,M_z, M_gz)
max = np.max(cross)

plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
plt.plot(E_cm_Arr, cross, label = r"$M_{tot}$")
plt.plot(compHepE,compHepCS, "--", label = "Comp Hep")
plt.plot(E_cm_Arr[np.where(cross == max)], max,  "x", markersize = 10, linewidth = 10, \
            label = f"Energy at peak:{E_cm_Arr[np.where(cross == max)][0]:.2f}")
plt.xlabel(r"$\sqrt{s} [Gev]$", fontsize=14)
plt.ylabel("Total cross section " + r"$\sigma [pb]$", fontsize=14)
plt.yscale("log")
plt.title("Cross section as a function of " + r"$\sqrt{s}$"+ " " + r"[$\mu^-,\mu^+ \rightarrow b,\bar{b}$]", fontsize = 14)
plt.legend(fontsize = 15)
plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.savefig("articles/totCross.pdf")
plt.show()

def asymmetry(*args):
    global E_cm, E, s, p, pp, c_g, c_z, c_gz
    asym = np.zeros(M)
    for i in range(M):
        #Update constants
        E_cm = E_cm_Arr[i]
        E = E_cm/2
        s = E_cm**2
        c_g = Q**2*8*e**4/(s**2)
        c_z = 8*g_z**4*1/(s-m_z**2)**2
        c_gz =  -Q*8*e**2*g_z**2 * 1/((s-m_z**2)*s)
        p = np.sqrt(E**2-m_m**2)
        pp = np.sqrt(E**2-m_b**2)

        if len(args)>1:
            cross = calcCrossSection(args[0], args[1], args[2])
        else:
            cross = calcCrossSection(args[0])

        sigma_1 = np.trapz(cross[int(N/2):])
        sigma_2 = np.trapz(cross[:int(N/2)])
        asym[i] = (sigma_1-sigma_2)/(sigma_2 + sigma_1)
    return asym


"""
m,M -> b,B
"""
data = np.loadtxt("data/Asymmetry_ub.txt", skiprows=3)
compHepE = data[:, 0]
compHepAs = data[:, 1]
Q = -1/3

plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')

plt.plot(E_cm_Arr, asymmetry(M_g,M_z, M_gz), label = r"$M_{tot}$")
plt.plot(E_cm_Arr, asymmetry(M_g), label = r"$M_{\gamma}$")
plt.plot(E_cm_Arr, asymmetry(M_z), label = r"$M_{z}$")
plt.plot(compHepE,compHepAs, "--", label = "Comp Hep")

plt.xlabel(r"$\sqrt{s} [Gev]$", fontsize=14)
plt.ylabel(r"$A_{FB}$", fontsize=14)
plt.title("Forward-Backward Asymmetry  "+r"$\mu^-,\mu^+ \rightarrow b,\bar{b}$", fontsize = 14)
plt.legend(fontsize = 15)
plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.savefig("articles/Asymmtry_classic.pdf")
plt.show()

"""
m,M -> c,C
"""
g_a = 0.19*0.5
g_b = 0.5*0.5
g_tot = g_ap*g_bp*g_a*g_b 
Omega_p =  (g_a**2 + g_b**2)
Omega_tp =  (g_a**2 - g_b**2)
m_b = 1.275#Gev
Q = 2/3

data = np.loadtxt("data/Asymmetry_cC.txt", skiprows=3)
compHepE = data[:, 0]
compHepAs = data[:, 1]

plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')

plt.plot(E_cm_Arr, asymmetry(M_g,M_z, M_gz), label = r"$M_{tot}$")
plt.plot(E_cm_Arr, asymmetry(M_g), label = r"$M_{\gamma}$")
plt.plot(E_cm_Arr, asymmetry(M_z), label = r"$M_{z}$")
plt.plot(compHepE,compHepAs, "--", label = "Comp Hep")

plt.xlabel(r"$\sqrt{s} [Gev]$", fontsize=14)
plt.ylabel(r"$A_{FB}$", fontsize=14)
plt.title("Forward-Backward Asymmetry  "+r"$\mu^-,\mu^+ \rightarrow c,\bar{c}$", fontsize = 14)
plt.legend(fontsize = 15)
plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.savefig("articles/Asymmtry_cC.pdf")
plt.show()

"""
m,M -> e,E
"""
g_a = -0.04*0.5
g_b = -0.5*0.5
g_tot = g_ap*g_bp*g_a*g_b 
Omega_p =  (g_a**2 + g_b**2)
Omega_tp =  (g_a**2 - g_b**2)
m_b = 0.5110 * 1e-3
Q = -1

data = np.loadtxt("data/Asymmetry_eE.txt", skiprows=3)
compHepE = data[:, 0]
compHepAs = data[:, 1]

plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')

plt.plot(E_cm_Arr, asymmetry(M_g,M_z, M_gz), label = r"$M_{tot}$")
plt.plot(E_cm_Arr, asymmetry(M_g), label = r"$M_{\gamma}$")
plt.plot(E_cm_Arr, asymmetry(M_z), label = r"$M_{z}$")
plt.plot(compHepE,compHepAs, "--", label = "Comp Hep")

plt.xlabel(r"$\sqrt{s} [Gev]$", fontsize=14)
plt.ylabel(r"$A_{FB}$", fontsize=14)
plt.title("Forward-Backward Asymmetry  "+r"$\mu^-,\mu^+ \rightarrow e^-,e^+$", fontsize = 14)
plt.legend(fontsize = 15)
plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.savefig("articles/Asymmtry_eE.pdf")
plt.show()


"""
Z-prime plots
"""
data = np.loadtxt("data/Zp_AS.txt", skiprows=3)
compHepE = data[:, 0]
compHepAS = data[:, 1]

plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')

plt.plot(compHepE, compHepAS, label = r"$Z'$")


plt.xlabel(r"$\sqrt{s} [Gev]$", fontsize=14)
plt.ylabel(r"$A_{FB}$", fontsize=14)
plt.title("Forward-Backward Asymmetry "+r"$\mu^-,\mu^+ \rightarrow \mu^-,\mu^+$" + " with z'", fontsize = 14)
plt.legend(fontsize = 15)
plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.savefig("articles/Asymmtry_Zp.pdf")
plt.show()

data = np.loadtxt("data/Zp_CS.txt", skiprows=3)
compHepE = data[:, 0]
compHepCS = data[:, 1]
max1 = np.max(compHepCS[:int(len(compHepCS)/2)])
max2 = np.max(compHepCS[int(len(compHepCS)/2):])
E1 = compHepE[np.where(compHepCS[:int(len(compHepCS)/2)] == max1)][0]
E2 = compHepE[int(len(compHepCS)/2):][np.where(compHepCS[int(len(compHepCS)/2):] == max2)][0]

plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')

plt.plot(compHepE, compHepCS, label = r"$Z'$")
plt.plot(E1, max1,  "x", markersize = 10, linewidth = 10, label = f"Energy: {E1:.2f}")
plt.plot(E2, max2,  "x", markersize = 10, linewidth = 10, label = f"Energy: {E2:.2f}")

plt.xlabel(r"$\sqrt{s} [Gev]$", fontsize=14)
plt.ylabel("Total cross section " + r"$\sigma [pb]$", fontsize=14)
plt.yscale("log")
plt.title("Total cross section with z'", fontsize = 14)
plt.legend(fontsize = 15)
plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.savefig("articles/Crossection_Zp.pdf")
plt.show()

data = np.loadtxt("data/Zp_DCS.txt", skiprows=3)
compHepE = data[:, 0]
compHepDCS = data[:, 1]

plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')

plt.plot(compHepE, compHepDCS, label = r"$Z'$")


plt.xlabel(r"$\cos(\theta)$", fontsize=15)
plt.ylabel(r"$d\sigma/dcos\theta$  [pb/rad]", fontsize=15)
plt.title("Differential cross section with z'", fontsize = 14)
plt.legend(fontsize = 15)
plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.savefig("articles/Crossection_DCS.pdf")
plt.show()