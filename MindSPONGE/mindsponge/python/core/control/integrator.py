from mindspore.ops import functional as F
import mindspore.nn as nn
import mindspore.numpy as mnp

class LeapFrog(nn.Cell):
    def __init__(self, space):
        super(LeapFrog, self).__init__()
        self.crd = space.coordinates
        self.vel = space.velocities
        self.dt = space.dt
        self.inverse_mass = space.inverse_mass

    def construct(self, force):
        crd = self.crd
        dt = self.dt
        vel = self.vel
        force = force * -1.0
        acc = mnp.expand_dims(self.inverse_mass, -1) * force
        vel = vel + dt * acc
        crd = crd + dt * vel
        return crd, vel

class LeapFrogLiuJian(nn.Cell):
    def __init__(self, space):
        super(LeapFrogLiuJian, self).__init__()
        self.crd = space.coordinates
        self.vel = space.velocities
        self.dt = space.dt
        self.inverse_mass = space.inverse_mass
        self.exp_gamma = space.exp_gamma

    def construct(self, force, random_force):
        crd = self.crd
        dt = self.dt
        half_dt = dt / 2.0
        vel = self.vel
        force = force * -1.0
        acc = mnp.expand_dims(self.inverse_mass, -1) * force
        vel = vel + dt * acc
        crd = crd + half_dt * vel
        vel = self.exp_gamma * vel + mnp.sqrt(mnp.expand_dims(self.inverse_mass, -1)) * random_force
        crd = crd + half_dt * vel
        return crd, vel

class GradientDescent(nn.Cell):
    def __init__(self, space):
        super(GradientDescent, self).__init__()
        self.crd = space.coordinates
        self.vel = space.velocities
        self.dt = space.dt

    def construct(self, force):
        crd = self.crd
        dt = self.dt
        crd = crd - dt * force
        return crd
