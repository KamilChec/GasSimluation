import numpy as np
import sys
import math
import random


class Argon:
    N = 0
    k = 8.31 * 10 ** (-3)
    T0 = 273

    def __init__(self, n=8, m=40, epsilon=1, R=0.38, f=10000, L=2.3, a=0.38, tau=0.002, s_o=100, s_d=2000):
        self.s_d = s_d
        self.s_o = s_o
        self.L = L
        self.tau = tau
        self.a = a
        self.f = f
        self.R = R
        self.epsilon = epsilon
        self.m = m
        self.n = n

    def read_parameters(self, file_name):
        with open(file_name, 'r') as inp:
            parameters = inp.read().splitlines()
        self.n = int(parameters[0])
        self.m = float(parameters[1])
        self.epsilon = int(parameters[2])
        self.R = float(parameters[3])
        self.f = int(parameters[4])
        self.L = float(parameters[5])
        self.a = float(parameters[6])
        self.tau = float(parameters[7])
        self.s_o = int(parameters[8])
        self.s_d = float(parameters[9])

    def print_particles(self, file, particles, first_use):
        if first_use:
            out_file = open(file, "w")
            out_file.write(str(self.N) + 2 * "\n")
        else:
            out_file = open(file, "a")
        for particle in particles:
            out_file.write("Ar " + str(particle.r[0]) + " " + str(particle.r[1]) + " " + str(particle.r[2]) + "\n")
        out_file.close()

    def start_position(self):
        particles = []
        b0 = [self.a, 0, 0]
        b1 = [self.a / 2., self.a * np.sqrt(3) / 2., 0]
        b2 = [self.a / 2., self.a * np.sqrt(3) / 6., self.a * np.sqrt(2. / 3.)]
        const = (self.n - 1) / 2.
        self.N = self.n ** 3
        for i0 in range(self.n):
            for i1 in range(self.n):
                for i2 in range(self.n):
                    r = np.dot(i0 - const, b0) + np.dot(i1 - const, b1) + np.dot(i2 - const, b2)
                    particles.append(Particle(r))
        p = []
        P = np.zeros([1, 3])
        for i in range(self.N):
            vec_p = []
            for q in range(3):
                kinetic_energy = -1 / 2. * self.k * self.T0 * math.log(random.random())
                vec_p.append((random.choice([-1, 1]) * 2 * math.sqrt(2 * self.m * kinetic_energy)))
            p.append(vec_p)
            P += vec_p
        for momentum, particle in zip(p, particles):
            particle.set_p(np.subtract(momentum, P / self.N))
        return particles

    def V_s(self, particle):
        normalised_r = np.linalg.norm(particle.r)
        if normalised_r >= self.L:
            return 1/2 * self.f * (normalised_r - self.L)**2
        else:
            return 0

    def F_s(self, particle):
        normalised_r = np.linalg.norm(particle.r)
        if normalised_r >= self.L:
            return np.dot(self.f * ((self.L - normalised_r) / normalised_r), particle.r)
        else:
            return 0

    def P_temp(self, F_s):
        return 1/(4 * math.pi * self.L**2) * np.linalg.norm(F_s)

    def V_p(self, particle_A, particle_B):
        normalised_rij = np.linalg.norm(np.subtract(particle_A.r, particle_B.r))
        return self.epsilon * ((self.R / normalised_rij)**12 - 2 * (self.R / normalised_rij)**6)

    def F_p(self, particle_A, particle_B):
        normalised_rij = np.linalg.norm(np.subtract(particle_A.r, particle_B.r))
        return np.dot(12 * self.epsilon * ((self.R/normalised_rij)**12 - 2 * (self.R/normalised_rij)**6) /
                      normalised_rij**2, np.subtract(particle_A.r, particle_B.r))

    def count_F_V_P(self, state):
        state.set_V(0)
        state.set_P(0)
        state.resetting_F(self.N)
        V = 0
        P = 0
        F = []
        for i in range(self.N):
            V += self.V_s(state.particles[i])
            F_s = self.F_s(state.particles[i])
            F.append(F_s)
            P += self.P_temp(F_s)
            for j in range(i):
                V += self.V_p(state.particles[i], state.particles[j])
                F[i] += self.F_p(state.particles[i], state.particles[j])
                F[j] -= self.F_p(state.particles[i], state.particles[j])
        print(np.shape(F))
        print(F[:3])
        print(V)
        print(P)
        state.set_V(V)
        state.set_P(P)
        state.set_F(F, self.N)

class State(Argon):
    def __init__(self, particles, V=0, P=0, H=0, T=0):
        self.T = T
        self.H = H
        self.P = P
        self.V = V
        self.particles = particles

    def set_V(self, V):
        self.V = V

    def set_P(self, P):
        self.P = P

    def set_F(self, F, N):
        self.F = F
        for i in range(N):
            self.particles[i].set_F(F[i])

    def resetting_F(self, N):
        for i in range(N):
            self.particles[i].set_F(np.zeros([1, 3]))

class Particle(State):
    def __init__(self, r, p=0, F=0):
        self.r = r
        self.p = p
        self.F = F

    def set_p(self, p):
        self.p = p

    def set_F(self, F):
        self.F = F

    def F(self):
        return [self.F[0], self.F[1], self.F[2]]

    def r(self):
        return [self.r[0], self.r[1], self.r[2]]


def main():
    try:
        sys.argv[1], sys.argv[2]
    except IndexError:
        print("insert files into arguments")
        return 0
    argon = Argon()
    argon.read_parameters(sys.argv[1])
    state = State(particles=argon.start_position())
    argon.print_particles(sys.argv[2], state.particles, first_use=True)
    argon.count_F_V_P(state)


if __name__ == "__main__":
    main()
