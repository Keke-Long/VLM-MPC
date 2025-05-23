import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

# MPC function
def mpc_fun(v_ego,lead_info, N,dt,Q,R,Q_h,tau, desired_speed, desired_headway):
    a_max = 2.5
    a_min = -3
    s0 = 4.2
    min_headway = 1
    slack_penalty = 1e5

    # optimization variable
    u = Variable((N,1))
    v = Variable((N+1,1))
    d = Variable((N+1,1))
    a_sigma = Variable((N,1), nonneg=True)

    # constrains
    constraints = [d[0]==0, v[0]==v_ego]
    for i in range(N):
        constraints +=[
            d[i+1] == d[i] + dt * v[i],
            v[i+1] == v[i] + dt * (u[i]/tau),
            v[i] <= desired_speed,
            v[i] >= 0,
            u[i] <= a_max + a_sigma[i],
            u[i] >= a_min - a_sigma[i],
        ]

    # cost function
    cost = sum([Q * (v[i] - desired_speed)**2 + R * u[i]**2 + slack_penalty * a_sigma[i] for i in range(N)])

    if lead_info:
        d0, lead_v = lead_info  #d0 is the distance between lead and ego vehicle
        for i in range(N):
            expected_distance = d0 + lead_v*i*dt - d[i] - desired_headway*v[i] - s0
            cost += Q_h * (expected_distance)**2

            constraints += [d0 + lead_v*i*dt - d[i] - min_headway*v[i] - s0 >= 0]

    prob = Problem(Minimize(cost), constraints)
    prob.solve()

    if prob.status !='optimal':
        raise ValueError('MPC not solved')

    return d.value.flatten(), v.value.flatten(), u.value.flatten()


def PID_fun(d_ego,d_lead,v_ego,desired_speed, desired_headway):
    kp_dist = 0.9
    kp_speed = 0.01

    # distance error
    desired_distance = desired_headway*v_ego
    distance_error = d_lead - d_ego - desired_distance

    speed_error = desired_speed - v_ego

    control = kp_dist*distance_error + kp_speed*speed_error #acceleration
    return control


# Parameters
dt = 0.5
time_steps = 50
tau = 0.5

N = 10
Q = 1
R = 1
Q_h = 2
desired_speed = 7
desired_headway = 2

# Define the trajectory of leading vehicle
lead_speeds = np.concatenate([
    np.full(10, 6),
    np.linspace(6, 4, 10),
    np.linspace(4, 7, 10),
    np.full(20, 5)
])
lead_position = np.cumsum(lead_speeds * dt) + 10


# Main simulation loop for MPC
v_ego = 4
x_ego = 0
v_ego_traj_mpc = [v_ego]
x_ego_traj_mpc = [x_ego]
for t in range(time_steps):
    d0 = lead_position[t] - x_ego
    lead_info = (d0, lead_speeds[t])

    d_control, v_control, u_control = mpc_fun(v_ego, lead_info, N,dt,Q,R,Q_h,tau, desired_speed, desired_headway)
    v_ego += u_control[0]*dt
    x_ego += v_ego *dt

    v_ego_traj_mpc.append(float(v_ego))
    x_ego_traj_mpc.append(float(x_ego))

# Main silulation loop for PID
v_ego = 4
x_ego = 0
v_ego_traj_pid = [v_ego]
x_ego_traj_pid = [x_ego]
for t in range(time_steps):
    u_pid = PID_fun(x_ego, lead_position[t], v_ego, desired_speed, desired_headway)
    v_ego += u_pid * dt
    x_ego += v_ego *dt

    v_ego_traj_pid.append(float(v_ego))
    x_ego_traj_pid.append(float(x_ego))


# plot
plt.figure(figsize=(10,6))
plt.plot(x_ego_traj_mpc, label = 'ego vehicle (MPC)', linestyle='--')
plt.plot(x_ego_traj_pid, label = 'ego vehicle (PID)', linestyle=':')
plt.plot(lead_position, label = 'lead vehicle')
plt.xlabel('Time(s)')
plt.ylabel('Position(m)')
plt.legend()
plt.show()