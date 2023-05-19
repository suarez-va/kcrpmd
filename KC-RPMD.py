import numpy as np

a=5e-6
b=400.
L=0.1

ms=1836.
omegas=2.28e-3
s0=-2.40
s1=2.40
epsilon=0.#-0.236
K=6.67e-7
M=1836.
omegac=2.28e-3
gamma=1.*M*omegac
fmodes=12
beta=1./(300*3.16681156345e-6)
my=K**2*beta**3/(2*np.pi*L**2) #times big number so y is fixed
eta=2*np.pi
gammay=1/(2*(2-L))*np.sqrt((1-2*np.log((2-L)/L*np.sqrt(a/np.pi)*eta*beta**2)-4*np.log(K))/(beta*my))

args = {"ms":ms,"omegas":omegas,"s0":s0,"s1":s1,"epsilon":epsilon,"K":K,"M":M,"omegac":omegac,"gamma":gamma,"fmodes":fmodes,"beta":beta,"my":my,"eta":eta,"a":a,"b":b,"L":L,"gammay":gammay}

##############################

def V0(s,x,args):
    (ms,omegas,s0,M,omegac,gamma,fmodes)=(args["ms"],args["omegas"],args["s0"],args["M"],args["omegac"],args["gamma"],args["fmodes"])
    omegaj=-omegac*np.log((np.arange(fmodes)+0.5)/fmodes); cj=omegaj*np.sqrt(2*gamma*M*omegac/(fmodes*np.pi))
    return 0.5*ms*omegas**2*(s-s0)**2+np.sum(0.5*M*omegaj**2*(x-cj*s/(M*omegaj**2))**2)

def V1(s,x,args):
    (ms,omegas,s1,epsilon,M,omegac,gamma,fmodes)=(args["ms"],args["omegas"],args["s1"],args["epsilon"],args["M"],args["omegac"],args["gamma"],args["fmodes"])
    omegaj=-omegac*np.log((np.arange(fmodes)+0.5)/fmodes); cj=omegaj*np.sqrt(2*gamma*M*omegac/(fmodes*np.pi))
    return 0.5*ms*omegas**2*(s-s1)**2+epsilon+np.sum(0.5*M*omegaj**2*(x-cj*s/(M*omegaj**2))**2)

def f(y,theta,args):
    b,L=args["b"],args["L"]
    if theta==-1 or theta==1:
        ltheta=2.-L
    elif theta==0:
        ltheta=L
    else:
        print("ERROR: theta should be -1, 0, or 1 !!! >:( ")
        ltheta=0.
    return 1/(ltheta*(1+np.exp(b*(2*abs(y-theta)-ltheta))))

def w(s,args):
    (ms,omegas,s0,s1,epsilon,K)=(args["ms"],args["omegas"],args["s0"],args["s1"],args["epsilon"],args["K"])
    return (-ms*omegas**2*(s0-s1)*s+0.5*ms*omegas**2*(s0**2-s1**2)-epsilon)/K

def phi(s,x,args):
    (K,beta)=(args["K"],args["beta"])
    return (2*np.cosh(beta/2*K*np.sqrt(w(s,args)**2+4))*np.exp(-beta/2*(V0(s,x,args)+V1(s,x,args)))-np.exp(-beta*V0(s,x,args))-np.exp(-beta*V1(s,x,args)))

def VKC(s,x,y,args):
    return -1/beta*np.log(np.sqrt(a/np.pi)*eta*f(y,0,args)*np.exp(-a*w(s,args)**2)*phi(s,x,args)+f(y,-1,args)*np.exp(-beta*V0(s,x,args))+f(y,1,args)*np.exp(-beta*V1(s,x,args)))

##############################

def F0s(s,x,args):
    (ms,omegas,s0,M,omegac,gamma,fmodes)=(args["ms"],args["omegas"],args["s0"],args["M"],args["omegac"],args["gamma"],args["fmodes"])
    omegaj=-omegac*np.log((np.arange(fmodes)+0.5)/fmodes); cj=omegaj*np.sqrt(2*gamma*M*omegac/(fmodes*np.pi))
    return -ms*omegas**2*(s-s0)+np.sum(cj*(x-cj*s/(M*omegaj**2)))

def F1s(s,x,args):
    (ms,omegas,s1,epsilon,M,omegac,gamma,fmodes)=(args["ms"],args["omegas"],args["s1"],args["epsilon"],args["M"],args["omegac"],args["gamma"],args["fmodes"])
    omegaj=-omegac*np.log((np.arange(fmodes)+0.5)/fmodes); cj=omegaj*np.sqrt(2*gamma*M*omegac/(fmodes*np.pi))
    return -ms*omegas**2*(s-s1)+np.sum(cj*(x-cj*s/(M*omegaj**2)))

def Fx(s,x,args):
    (M,omegac,gamma,fmodes)=(args["M"],args["omegac"],args["gamma"],args["fmodes"])
    omegaj=-omegac*np.log((np.arange(fmodes)+0.5)/fmodes); cj=omegaj*np.sqrt(2*gamma*M*omegac/(fmodes*np.pi))
    return -M*omegaj**2*(x-cj*s/(M*omegaj**2))

def dphis(s,x,args):
    (ms,omegas,s0,s1,K,beta)=(args["ms"],args["omegas"],args["s0"],args["s1"],args["K"],args["beta"])
    return (beta*((F0s(s,x,args)+F1s(s,x,args))*np.cosh(beta/2*K*np.sqrt(w(s,args)**2+4))-(ms*omegas**2*(s0-s1)*w(s,args)/np.sqrt(w(s,args)**2+4))*np.sinh(beta/2*K*np.sqrt(w(s,args)**2+4)))*np.exp(-beta/2*(V0(s,x,args)+V1(s,x,args)))-beta*F0s(s,x,args)*np.exp(-beta*V0(s,x,args))-beta*F1s(s,x,args)*np.exp(-beta*V1(s,x,args)))

def dphix(s,x,args):
    (ms,omegas,s0,s1,K,beta)=(args["ms"],args["omegas"],args["s0"],args["s1"],args["K"],args["beta"])
    return (2*beta*Fx(s,x,args)*np.cosh(beta/2*K*np.sqrt(w(s,args)**2+4))*np.exp(-beta/2*(V0(s,x,args)+V1(s,x,args)))-beta*Fx(s,x,args)*np.exp(-beta*V0(s,x,args))-beta*Fx(s,x,args)*np.exp(-beta*V1(s,x,args)))

def FKCs(s,x,y,args):
    (ms,omegas,s0,s1,K,beta,eta,a)=(args["ms"],args["omegas"],args["s0"],args["s1"],args["K"],args["beta"],args["eta"],args["a"])
    return ((np.sqrt(a/np.pi)*eta*f(y,0,args)*(2*a*w(s,args)*phi(s,x,args)*ms*omegas**2*(s0-s1)/K+dphis(s,x,args))*np.exp(-a*w(s,args)**2)+beta*f(y,-1,args)*F0s(s,x,args)*np.exp(-beta*V0(s,x,args))+beta*f(y,1,args)*F1s(s,x,args)*np.exp(-beta*V1(s,x,args)))/(beta*(np.sqrt(a/np.pi)*eta*f(y,0,args)*np.exp(-a*w(s,args)**2)*phi(s,x,args)+f(y,-1,args)*np.exp(-beta*V0(s,x,args))+f(y,1,args)*np.exp(-beta*V1(s,x,args)))))

def FKCx(s,x,y,args):
    (beta,eta,a)=(args["beta"],args["eta"],args["a"])
    return ((np.sqrt(a/np.pi)*eta*f(y,0,args)*dphix(s,x,args)*np.exp(-a*w(s,args)**2)+beta*f(y,-1,args)*Fx(s,x,args)*np.exp(-beta*V0(s,x,args))+beta*f(y,1,args)*Fx(s,x,args)*np.exp(-beta*V1(s,x,args)))/(beta*(np.sqrt(a/np.pi)*eta*f(y,0,args)*np.exp(-a*w(s,args)**2)*phi(s,x,args)+f(y,-1,args)*np.exp(-beta*V0(s,x,args))+f(y,1,args)*np.exp(-beta*V1(s,x,args)))))

def df(y,theta,args):
    b,L=args["b"],args["L"]
    if theta==0:
        ltheta=L
    else:
        ltheta=2.-L
    if abs(b*(abs(y-theta)-ltheta/2.))>200.:
        return 0.
    elif y>theta:
        return -b/(2*ltheta*np.cosh(b*(abs(y-theta)-ltheta/2.))**2)
    else:
        return b/(2*ltheta*np.cosh(b*(abs(y-theta)-ltheta/2.))**2)

def FKCy(s,x,y,args):
    (beta,eta,a)=(args["beta"],args["eta"],args["a"])
    return ((np.sqrt(a/np.pi)*eta*df(y,0,args)*np.exp(-a*w(s,args)**2)*phi(s,x,args)+df(y,-1,args)*np.exp(-beta*V0(s,x,args))+df(y,1,args)*np.exp(-beta*V1(s,x,args)))/(beta*(np.sqrt(a/np.pi)*eta*f(y,0,args)*np.exp(-a*w(s,args)**2)*phi(s,x,args)+f(y,-1,args)*np.exp(-beta*V0(s,x,args))+f(y,1,args)*np.exp(-beta*V1(s,x,args)))))

##############################

def positiony(y,vy,dt):
    return y+vy*dt

def velocityy(vy,Fy,dt,args):
    (my)=(args["my"])
    return vy+Fy*dt/(2*my)

def Langevin(args):
    (beta,my,gammay)=(args["beta"],args["my"],args["gammay"])
    sigma=np.sqrt(2*gammay/(beta*my))
    xi=np.random.normal(); theta=np.random.normal()
    return sigma,xi,theta

def positionyLangevin(y,vy,dt,langevin):
    sigma,xi,theta=langevin
    return y+vy*dt+sigma/(2*np.sqrt(3))*theta*np.sqrt(dt**3)

def velocityyLangevin(v,Fy,dt,args,langevin):
    sigma,xi,theta=langevin
    (my,gammay)=(args["my"],args["gammay"])
    return (v+Fy*dt/(2*my)-0.5*gammay*v*dt+0.5*sigma*xi*np.sqrt(dt)-1/8**gammay*dt**2*(F/my-gammay*v)-0.25*gammay*sigma*np.sqrt(dt**3)*(xi/2+theta/np.sqrt(3)))

def positions(s,vs,dt):
    return s+vs*dt

def velocitys(vs,Fs,dt,args):
    (ms)=(args["ms"])
    return vs+Fs*dt/(2*ms)

def boltzmanns(args):
    (ms,beta)=(args["ms"],args["beta"])
    return np.random.normal(scale=1/np.sqrt(beta*ms))

def positionx(x,vx,dt):
    return x+vx*dt

def velocityx(vx,Fx,dt,args):
    (M)=(args["M"])
    return vx+Fx*dt/(2*M)

def boltzmannx(args):
    (M,fmodes,beta)=(args["M"],args["fmodes"],args["beta"])
    return np.random.normal(scale=1/np.sqrt(beta*M),size=(fmodes))

def time(t,dt):
    return t+dt

################################################################################################################################

file_output1 = open('output.dat', 'w')
file_output2 = open('output.positions.dat', 'w')
file_output3 = open('output.init.dat', 'w')

t=0.
tf=100000000.
dt=2.5
Nprint=2500
Nresample=25000

y=0.
s=0.5*(s0+s1)-epsilon/(ms*omegas**2*(s0-s1))
x=np.zeros((fmodes))

vy=0.
vs=boltzmanns(args)
vx=boltzmannx(args)

total_steps=int(np.ceil((tf-t)/dt))

Forces=FKCs(s,x,y,args);Forcex=FKCx(s,x,y,args);Forcey=FKCy(s,x,y,args)

output3 = np.zeros(fmodes+2)
output3[0]=vy
output3[1]=vs
output3[2:]=vx
np.savetxt(file_output3, output3.reshape(1, output3.shape[0]), '%20.7e')#file_output.write(f"\n time: {t} \n Electronic: {y} \n Nuclear: \n")
file_output3.flush()

for step in range(total_steps):
    if (np.mod(step,Nresample)==0):
        vs=boltzmanns(args)
        vx=boltzmannx(args)
    if (np.mod(step,Nprint)==0):
        PE=VKC(s,x,y,args)
        KE=0.5*ms*vs**2+0.5*M*np.sum(vx**2)+0.5*my*vy**2
        TE=PE+KE
        #print(f"Printing data at time step: {t}")
        output1 = np.zeros(4)
        output1[0]=t
        output1[1]=KE
        output1[2]=PE
        output1[3]=TE
        np.savetxt(file_output1, output1.reshape(1, output1.shape[0]), '%20.7e')#file_output.write(f"\n time: {t} \n Electronic: {y} \n Nuclear: \n")
        file_output1.flush()
        output2 = np.zeros(fmodes+3)
        output2[0]=t
        output2[1]=y
        output2[2]=s
        output2[3:]=x
        np.savetxt(file_output2, output2.reshape(1, output2.shape[0]), '%20.7e')#file_output.write(f"\n time: {t} \n Electronic: {y} \n Nuclear: \n")
        file_output2.flush()
    #langevin=Langevin(args)
    vs=velocitys(vs,Forces,dt,args);vx=velocityx(vx,Forcex,dt,args);vy=0.#vy=velocityy(vy,Forcey,dt,args)#vy=velocityyLangevin(vy,Fy,dt,args,langevin)
    s=positions(s,vs,dt);x=positionx(x,vx,dt);y=0.#y=positiony(y,vy,dt)#y=positionyLangevin(y,vy,dt,langevin)
    Forces=FKCs(s,x,y,args);Forcex=FKCx(s,x,y,args);Forcey=0.#Forcey=FKCy(s,x,y,args)
    vs=velocitys(vs,Forces,dt,args);vx=velocityx(vx,Forcex,dt,args);vy=0.#vy=velocityy(vy,Forcey,dt,args)#vy=velocityyLangevin(vy,Fy,dt,args,langevin)
    t=time(t,dt)
