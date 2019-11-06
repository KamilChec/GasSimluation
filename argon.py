import numpy as np
import sys
import math
import random
from tqdm.auto import tqdm


class Argon:
    N = 0
    k = 8.31 * 10**(-3)
    T0 = 273
    s_out = 10
    s_xyz = 10

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
            out_file.write(str(self.N) + 2 * "\n")
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
                    r = np.asarray(np.dot(i0 - const, b0) + np.dot(i1 - const, b1) + np.dot(i2 - const, b2))
                    particles.append(Particle(r))
        p = []
        P = [0, 0, 0]
        for i in range(self.N):
            vec_p = [0, 0, 0]
            for q in range(3):
                kinetic_energy = -0.5 * self.k * self.T0 * math.log(random.random())
                vec_p[q] = (random.choice([-1, 1]) * math.sqrt(2 * self.m * kinetic_energy))
            p.append(vec_p)
            P += np.asarray(vec_p)
        for momentum, particle in zip(p, particles):
            particle.set_p(np.subtract(momentum, P / self.N))
        return particles

    def V_s(self, particle):
        normalised_r = np.linalg.norm(particle.r)
        if normalised_r >= self.L:
            return 0.5 * self.f * (normalised_r - self.L)**2
        else:
            return 0

    def F_s(self, particle):
        normalised_r = np.linalg.norm(particle.r)
        if normalised_r >= self.L:
            return np.dot(self.f * ((self.L - normalised_r) / normalised_r), particle.r)
        else:
            return 0

    def P_temp(self, F_s):
        return 1 / (4 * math.pi * self.L**2) * np.linalg.norm(F_s)

    def V_p(self, particle_A, particle_B):
        normalised_rij = np.linalg.norm(np.subtract(particle_A.r, particle_B.r))
        return self.epsilon * ((self.R/normalised_rij)**12 - 2 * (self.R/normalised_rij)**6)

    def F_p(self, particle_A, particle_B):
        normalised_rij = np.linalg.norm(np.subtract(particle_A.r, particle_B.r))
        return np.dot(12 * self.epsilon * ((self.R/normalised_rij)**12 - (self.R/normalised_rij)**6) /
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
        state.set_V(V)
        state.set_P(P)
        for i in range(self.N):
            state.particles[i].set_F(F[i])

    def count_T(self, state):
        kinetic_energy = 0
        for i in range(self.N):
            kinetic_energy += (np.linalg.norm(state.particles[i].p))**2 / (2 * self.m)
        return 2. / (3 * self.N * self.k) * kinetic_energy

    def count_H(self, state):
        kinetic_energy = 0
        for i in range(self.N):
            kinetic_energy += (np.linalg.norm(state.particles[i].p)) ** 2 / (2 * self.m)
        return kinetic_energy + state.V

    def save_state(self, file, state, first_use):
        if first_use:
            out_file = open(file, "w")
            out_file.write("Energy" + '\t' + "Potential" + '\t' + "Temperature" + '\t' + "Pressure" + "\n")
            out_file.write(str(state.H) + '\t' + str(state.V) + '\t' + str(state.T) + '\t' + str(state.P) + "\n")
        else:
            out_file = open(file, "a")
            out_file.write(str(state.H) + '\t' + str(state.V) + '\t' + str(state.T) + '\t' + str(state.P) + "\n")
        out_file.close()

    def simulation(self, state):
        average_H = average_T = average_P = 0
        for s in tqdm(range(int(self.s_o + self.s_d))):
            print("", end='\r')
            for i in range(self.N):
                p_halfTau = np.add(state.particles[i].p, np.dot((self.tau * 0.5), state.particles[i].F))
                r = np.add(state.particles[i].r, np.dot(self.tau / self.m, p_halfTau))
                state.particles[i].set_r(r)
            self.count_F_V_P(state)
            for i in range(self.N):
                state.particles[i].set_p(np.add(p_halfTau, np.dot(self.tau * 0.5, state.particles[i].F)))
            state.set_T(self.count_T(state))
            state.set_H(self.count_H(state))
            if s % self.s_out == 0:
                if s == 0:
                    self.save_state("state.txt", state, first_use=True)
                else:
                    self.save_state("state.txt", state, first_use=False)
            if s % self.s_xyz == 0:
                self.print_particles(sys.argv[2], state.particles, first_use=False)
            if s >= self.s_o:
                average_T += state.T
                average_H += state.H
                average_P += state.P
        average_P /= (self.s_o + self.s_d)
        average_H /= (self.s_o + self.s_d)
        average_T /= (self.s_o + self.s_d)
        print("Average value of:" + '\n' + "Energy: " + str(average_H) + '\t' + "Pressure: " + str(
            average_P) + '\t' + "Temperature: " + str(average_T))


class State(Argon):
    def __init__(self, particles, V=0, P=0, H=0, T=0):
        self.T = T
        self.H = H
        self.P = P
        self.V = V
        self.particles = particles

    def set_T(self, T):
        self.T = T

    def set_H(self, H):
        self.H = H

    def set_V(self, V):
        self.V = V

    def set_P(self, P):
        self.P = P

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

    def set_r(self, r):
        self.r = r

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
    argon.simulation(state)


if __name__ == "__main__":
    main()
