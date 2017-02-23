import numpy as np

# ### Runge-Kutta 4: dt step
def rk4_step(x, f, dt):
    a = f(x)
    b = f(x+0.5*dt*a)
    c = f(x+0.5*dt*b)
    d = f(x+dt*c)
    return x+dt*(a+2*b+2*c+d)/6.0


# ### Planetary motion: f and E
def bmg(x):
    tmp = -1.0/(x[0]**2+x[1]**2)**(1.5)
    return np.array([x[2], x[3], x[0]*tmp, x[1]*tmp], dtype=np.double)

def E(x):
    return 0.5*(x[2]**2+x[3]**2)-1.0/np.sqrt(x[0]**2+x[1]**2)

# ### Initial values
dt   = 0.01
t0   = 0
tmax = 100
x0 = np.array([0, 1.0/np.sqrt(10), 2.4, 0], dtype=np.double)
E0 = E(x0)

# ### Time evolution
t = t0
x = x0
solution = [[t, x[0], x[1], x[2], x[3], E(x)-E0]]
while True:
    x = rk4_step(x, bmg, dt)
    t = t+dt
    solution.append([t, x[0], x[1], x[2], x[3], E(x)-E0])
    if t>=tmax:
        break

# ### Saving solution
f = open("clbjw0_bead1_sol.txt", "w")
for line in solution:
    f.write(" ".join(map(str, line))+"\n")
f.close()