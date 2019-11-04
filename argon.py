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


class State(Argon):
    def __init__(self, V, P, H, T):
        self.T = T
        self.H = H
        self.P = P
        self.V = V


class Particle(State):
    def __init__(self, r, p=0):
        self.r = r
        self.p = p

    def set_p(self, p):
        self.p = p

    def r(self):
        return [self.x, self.y, self.z]


def main():
    try:
        sys.argv[1], sys.argv[2]
    except IndexError as error:
        print("insert arguments")
    argon = Argon()
    argon.read_parameters(sys.argv[1])
    particles = argon.start_position()
    argon.print_particles(sys.argv[2], particles, first_use=True)


if __name__ == "__main__":
    main()
