#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import print_function
from collections import deque
import copy
import matplotlib.pyplot as plt
import numpy as np
import serial
import torch
import lidar_to_grid_map as lg
#import PPO
import time
#import wiringpi
import multiprocessing
import random
import math
import icp
from NDT import ndt_match
from NDT import apply_transformation
# o-------------x(right)
# |
# |
# |
# |
# |
# y(up)(init_direction)
PWM1 = 15
A1 = 14
PWM2 = 10
B1 = 9
raw_msg = []

forward_dist = 120
backward_dist = 120
left_dist = 130
right_dist = 125


def get_angle_distance(target_angle1, target_angle2, conn):
    global raw_msg
    raw_msg = conn.recv()
    raw_tmp = raw_msg
    angle_dist1 = []
    angle_dist2 = []
    while len(angle_dist1) <= 100:
        for angle, distance in raw_tmp:
            if angle >= ((target_angle1 - 1) % 360) and angle <= (target_angle1 + 1) % 360:
                angle_dist1.append(distance)
    while len(angle_dist2) <= 100:
        for angle, distance in raw_tmp:
            if angle >= ((target_angle2 - 1) % 360) and angle <= (target_angle2 + 1) % 360:
                angle_dist2.append(distance)
    return [(sum(angle_dist1) / len(angle_dist1)), (sum(angle_dist2) / len(angle_dist2))]


def get_raw_info(conn):
    ser = serial.Serial("COM8", 230400, timeout=5)
    if ser.is_open == False:
        ser.open()
    msg = []
    while True:
        data = ser.read(2)  # 读取2个字节数据
        if data[0] == 0x54 and data[1] == 0x2C:  # 判断是否为数据帧头
            data = ser.read(45)  # 是数据帧头就读取整一帧，去掉帧头之后为45个字节
            start_angle = ((data[3] * 256 + data[2]) / 100.0 + 1) % 360
            end_angle = ((data[41] * 256 + data[40]) / 100.0 + 1) % 360
            angle = start_angle
            if end_angle > start_angle:
                step = (end_angle - start_angle) / 11
            else:
                step = (end_angle + 360 - start_angle) / 11
            for x in range(4, 40, 3):  # 2个字节的距离数据，1个信号强度数据，步长为3
                msg.append([angle, data[x + 1] * 256 + data[x]])
                angle = angle + step  # 角分辨率

        if len(msg) >= 4320:
            conn.send(msg)
            msg = []


def get_state(direction, now, end, conn):
    state = []
    threshold = 270
    raw_threshold = 150
    up_dist = []
    down_dist = []
    left_dist = []
    right_dist = []
    global raw_msg
    raw_msg = conn.recv()
    # print(raw_msg)
    # print(len(raw_msg))
    for angle, distance in raw_msg:
        if angle >= 325 or angle <= 35:
            up_dist.append(distance)
        elif angle >= 55 and angle <= 125:
            left_dist.append(distance)
        elif angle >= 145 and angle <= 215:
            down_dist.append(distance)
        elif angle > 235 and angle <= 305:
            right_dist.append(distance)
    up_avg = sum(up_dist) / len(up_dist)
    down_avg = sum(down_dist) / len(down_dist)
    left_avg = sum(left_dist) / len(left_dist)
    right_avg = sum(right_dist) / len(right_dist)

    raw_state = []

    if up_avg >= raw_threshold:
        raw_state.append(0)
    else:
        raw_state.append(1)
    if down_avg >= raw_threshold:
        raw_state.append(0)
    else:
        raw_state.append(1)
    if right_avg >= raw_threshold:
        raw_state.append(0)
    else:
        raw_state.append(1)
    if left_avg >= raw_threshold:
        raw_state.append(0)
    else:
        raw_state.append(1)

    if direction == '+y':
        if down_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if up_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if left_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if right_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
    elif direction == '+x':
        if right_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if left_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if down_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if up_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
    elif direction == '-x':
        if left_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if right_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if up_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if down_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
    else:
        if up_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if down_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if right_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
        if left_avg >= threshold:
            state.append(0)
        else:
            state.append(1)
    state = state + now
    state = state + end
    return raw_state, torch.FloatTensor(state)


def forward(distance):
    # wiringpi.pwmSetRange(128)
    wiringpi.softPwmCreate(PWM1, 0, 25)
    wiringpi.softPwmCreate(PWM2, 0, 25)
    wiringpi.digitalWrite(A1, 1)
    wiringpi.digitalWrite(B1, 0)
    wiringpi.softPwmWrite(PWM1, 20)
    wiringpi.softPwmWrite(PWM2, 6)
    i = 0
    while True:
        i = i + 1
        if i == distance:
            wiringpi.softPwmWrite(PWM1, 25)
            wiringpi.softPwmWrite(PWM2, 0)
            return
        time.sleep(0.01)


def backward(distance):
    # wiringpi.pwmSetRange(128)
    wiringpi.softPwmCreate(PWM1, 0, 25)
    wiringpi.softPwmCreate(PWM2, 0, 25)
    wiringpi.digitalWrite(A1, 0)
    wiringpi.digitalWrite(B1, 1)
    wiringpi.softPwmWrite(PWM1, 5)
    wiringpi.softPwmWrite(PWM2, 19)
    i = 0
    while True:
        i = i + 1
        if i == distance:
            wiringpi.softPwmWrite(PWM2, 25)
            wiringpi.softPwmWrite(PWM1, 0)
            return
        time.sleep(0.01)


def left(time1, speed=5):
    wiringpi.softPwmCreate(PWM2, 0, 25)
    wiringpi.softPwmCreate(PWM1, 0, 25)
    wiringpi.digitalWrite(A1, 0)
    wiringpi.digitalWrite(B1, 0)
    wiringpi.softPwmWrite(PWM1, speed)
    wiringpi.softPwmWrite(PWM2, speed)
    i = 0
    while True:
        if i == time1:
            wiringpi.softPwmWrite(PWM2, 0)
            wiringpi.softPwmWrite(PWM1, 0)
            return
        i = i + 1
        time.sleep(0.01)


def right(time1, speed=5):
    wiringpi.softPwmCreate(PWM1, 0, 25)
    wiringpi.softPwmCreate(PWM2, 0, 25)
    wiringpi.digitalWrite(A1, 1)
    wiringpi.digitalWrite(B1, 1)
    wiringpi.softPwmWrite(PWM1, 25 - speed)
    wiringpi.softPwmWrite(PWM2, 25 - speed)
    i = 0
    while True:
        if i == time1:
            wiringpi.softPwmWrite(PWM2, 25)
            wiringpi.softPwmWrite(PWM1, 25)
            return
        i = i + 1
        time.sleep(0.01)


def align_up(conn):
    L, R = get_angle_distance(343, 17, conn)
    print("L ={}".format(L))
    print("R ={}".format(R))
    time.sleep(1)
    L1, R1 = get_angle_distance(343, 17, conn)
    while abs(L1 - L) >= 1 or abs(R1 - R) >= 1:
        time.sleep(0.5)
        L, R = get_angle_distance(343, 17, conn)
        time.sleep(0.5)
        L1, R1 = get_angle_distance(343, 17, conn)
    threshold = 5
    # return
    if (L - R) >= threshold or (R - L) >= threshold:
        while True:
            if L - R >= threshold:
                right(20, 1)
            elif (R - L) >= threshold:
                left(20, 1)

            time.sleep(0.5)
            L1, R1 = get_angle_distance(343, 17, conn)
            time.sleep(0.5)
            L2, R2 = get_angle_distance(343, 17, conn)
            while abs(L1 - L2) >= 1 or abs(R1 - R2) >= 1:
                time.sleep(0.5)
                L1, R1 = get_angle_distance(343, 17, conn)
                time.sleep(0.5)
                L2, R2 = get_angle_distance(343, 17, conn)
            print("L ={}".format(L1))
            print("R ={}".format(R1))
            if abs(L1 - R1) <= threshold:
                break


def align_down(conn):
    L, R = get_angle_distance(197, 163, conn)
    print("L ={}".format(L))
    print("R ={}".format(R))
    time.sleep(1)
    L1, R1 = get_angle_distance(197, 163, conn)
    while abs(L1 - L) >= 1 or abs(R1 - R) >= 1:
        time.sleep(0.5)
        L, R = get_angle_distance(197, 163, conn)
        time.sleep(0.5)
        L1, R1 = get_angle_distance(197, 163, conn)
    threshold = 5
    # return
    if (L - R) >= threshold or (R - L) >= threshold:
        while True:
            if L - R >= threshold:
                left(20, 1)
            elif (R - L) >= threshold:
                right(20, 1)

            time.sleep(0.5)
            L1, R1 = get_angle_distance(197, 163, conn)
            time.sleep(0.5)
            L2, R2 = get_angle_distance(197, 163, conn)
            while abs(L1 - L2) >= 1 or abs(R1 - R2) >= 1:
                time.sleep(0.5)
                L1, R1 = get_angle_distance(197, 163, conn)
                time.sleep(0.5)
                L2, R2 = get_angle_distance(197, 163, conn)
            print("L ={}".format(L1))
            print("R ={}".format(R1))
            if abs(L1 - R1) <= threshold:
                break


def align_left(conn):
    L, R = get_angle_distance(287, 253, conn)
    print("L ={}".format(L))
    print("R ={}".format(R))
    time.sleep(1)
    L1, R1 = get_angle_distance(287, 253, conn)
    while abs(L1 - L) >= 1 or abs(R1 - R) >= 1:
        time.sleep(0.5)
        L, R = get_angle_distance(287, 253, conn)
        time.sleep(0.5)
        L1, R1 = get_angle_distance(287, 253, conn)
    threshold = 5
    # return
    if (L - R) >= threshold or (R - L) >= threshold:
        while True:
            if L - R >= threshold:
                left(20, 1)
            elif (R - L) >= threshold:
                right(20, 1)

            time.sleep(0.5)
            L1, R1 = get_angle_distance(287, 253, conn)
            time.sleep(0.5)
            L2, R2 = get_angle_distance(287, 253, conn)
            while abs(L1 - L2) >= 1 or abs(R1 - R2) >= 1:
                time.sleep(0.5)
                L1, R1 = get_angle_distance(287, 253, conn)
                time.sleep(0.5)
                L2, R2 = get_angle_distance(287, 253, conn)
            print("L ={}".format(L1))
            print("R ={}".format(R1))
            if abs(L1 - R1) <= threshold:
                break


def align_right(conn):
    L, R = get_angle_distance(73, 107, conn)
    print("L ={}".format(L))
    print("R ={}".format(R))
    time.sleep(1)
    L1, R1 = get_angle_distance(73, 107, conn)
    while abs(L1 - L) >= 1 or abs(R1 - R) >= 1:
        time.sleep(0.5)
        L, R = get_angle_distance(73, 107, conn)
        time.sleep(0.5)
        L1, R1 = get_angle_distance(73, 107, conn)
    threshold = 5
    # return
    if (L - R) >= threshold or (R - L) >= threshold:
        while True:
            if L - R >= threshold:
                right(20, 1)
            elif (R - L) >= threshold:
                left(20, 1)

            time.sleep(0.5)
            L1, R1 = get_angle_distance(73, 107, conn)
            time.sleep(0.5)
            L2, R2 = get_angle_distance(73, 107, conn)
            while abs(L1 - L2) >= 1 or abs(R1 - R2) >= 1:
                time.sleep(0.5)
                L1, R1 = get_angle_distance(73, 107, conn)
                time.sleep(0.5)
                L2, R2 = get_angle_distance(73, 107, conn)
            print("L ={}".format(L1))
            print("R ={}".format(R1))
            if abs(L1 - R1) <= threshold:
                break


def align(raw_state, conn):
    # raw_state = get_raw_info(conn) # in the order of up, down, left, right
    if sum(raw_state) == 0:
        print("Didn't find any walls!")
        return  # no wall to align, do nothing
    else:
        if raw_state[0] == 1:
            align_up(conn)
        elif raw_state[1] == 1:
            align_down(conn)
        elif raw_state[2] == 1:
            align_left(conn)
        else:
            align_right(conn)


def choose_action(pi):
    p = random.random()
    if p <= pi[0]:
        return 'up'
    elif p <= pi[0] + pi[1]:
        return 'down'
    elif p <= pi[0] + pi[1] + pi[2]:
        return 'left'
    else:
        return 'right'


def do_action(now, direction, action):
    next_direction = direction
    if action == 'None':
        return now, direction
    if direction == '+y':
        if action == 'up':
            forward(forward_dist)
        elif action == 'down':
            backward(backward_dist)
        elif action == 'left':
            right(right_dist)
            time.sleep(1)
            forward(forward_dist)
            next_direction = '-x'
        else:
            left(left_dist)
            time.sleep(1)
            forward(forward_dist)
            next_direction = '+x'
    elif direction == '-y':
        if action == 'up':
            backward(backward_dist)
        elif action == 'down':
            forward(forward_dist)
        elif action == 'left':
            left(left_dist)
            time.sleep(1)
            forward(forward_dist)
            next_direction = '-x'
        else:
            right(right_dist)
            time.sleep(1)
            forward(forward_dist)
            next_direction = '+x'
    elif direction == '+x':
        if action == 'up':
            right(right_dist)
            time.sleep(1)
            forward(forward_dist)
            next_direction = '+y'
        elif action == 'down':
            left(left_dist)
            time.sleep(1)
            forward(forward_dist)
            next_direction = '-y'
        elif action == 'left':
            backward(backward_dist)
        else:
            forward(forward_dist)
    else:
        if action == 'up':
            left(left_dist)
            time.sleep(1)
            forward(forward_dist)
            next_direction = '+y'
        elif action == 'down':
            right(right_dist)
            time.sleep(1)
            forward(forward_dist)
            next_direction = '-y'
        elif action == 'left':
            forward(forward_dist)
        else:
            backward(backward_dist)

    [x, y] = now
    if action == 'up':
        next = [x, y + 1]
    elif action == 'down':
        next = [x, y - 1]
    elif action == 'left':
        next = [x - 1, y]
    else:
        next = [x + 1, y]
    return next, next_direction


def collision_detect(state, action):
    if (state[0] == 1 and action == 'down') or (state[1] == 1 and action == 'up') or (
            state[2] == 1 and action == 'left') or (state[3] == 1 and action == 'right'):
        return True
    else:
        return False


def make_grid(grid_size, pos, grid, conn): # grid_size: mm; grid: np.array, 1 is wall, 0 is undefined, -1 is free
    # global raw_msg
    # raw_msg = conn.recv()
    # x_car, y_car = pos
    # list_x = []
    # list_y = []
    # hit_times = {}
    # new_grid = copy.deepcopy(grid)
    # s_x, s_y = new_grid.shape
    # for angle, distance in raw_msg:
    #     x = int((distance * math.sin(angle / 360 * 2 * math.pi) + grid_size / 2) // grid_size)
    #     y = int((distance * math.cos(angle / 360 * 2 * math.pi) + grid_size / 2) // grid_size)
    #     if x_car + x > 0 and x_car + x < s_x and y_car + y > 0 and y_car + y < s_y:
    #         list_x.append(x)
    #         list_y.append(y)
    #         if hit_times.get(tuple([x,y])) is None:
    #             hit_times[tuple([x, y])] = 1
    #         else:
    #             hit_times[tuple([x,y])] = hit_times[tuple([x,y])] + 1
    # list_hit_times = list(hit_times.items())
    # for (x,y), hit_times in list_hit_times:
    #     if hit_times >= 1 and grid[x,y] == 0:
    #         new_grid[x_car + x, y_car + y] = 1
    #
    # fringe = deque()
    # fringe.appendleft(tuple(pos))
    # while fringe:
    #     n = fringe.pop()
    #     n_x, n_y = n
    #     if n_x > 0:
    #         if new_grid[n_x - 1, n_y] == 0:
    #             new_grid[n_x - 1, n_y] = -1
    #             fringe.append(tuple([n_x - 1, n_y]))
    #     if n_x < s_x - 1:
    #         if new_grid[n_x + 1, n_y] == 0:
    #             new_grid[n_x + 1, n_y] = -1
    #             fringe.append(tuple([n_x + 1, n_y]))
    #     if n_y > 0:
    #         if new_grid[n_x, n_y - 1] == 0:
    #             new_grid[n_x, n_y - 1] = -1
    #             fringe.append(tuple([n_x, n_y - 1]))
    #     if n_y < s_y - 1:
    #         if new_grid[n_x, n_y + 1] == 0:
    #             new_grid[n_x, n_y + 1] = -1
    #             fringe.append(tuple([n_x, n_y + 1]))
    # return new_grid
    xyreso = 0.02
    global raw_msg
    raw_msg = conn.recv()
    raw_msg = conn.recv()
    raw_msg = conn.recv()
    raw_msg = conn.recv()
    angle = []
    distance = []
    for a,d in raw_msg:
        if abs(d) <= 1000 and d != 0:
            angle.append(math.radians(a))
            distance.append(d/1000)

    angle.append(0)
    distance.append(0)
    ox = np.sin(angle) * distance
    oy = np.cos(angle) * distance

    pmap, minx, maxx, miny, maxy, xyreso = lg.generate_ray_casting_grid_map(ox, oy, xyreso, True)
    xyres = np.array(pmap).shape
    #plt.figure(figsize=(20, 8))
    plt.imshow(pmap, cmap="PiYG_r")
    plt.clim(-1, 1)
    plt.gca().set_xticks(np.arange(-.5, xyres[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, xyres[0], 1), minor=True)
    plt.grid(True, which="minor", color="w", linewidth=.6, alpha=0.5)
    plt.colorbar()
    plt.show()
    return pmap

# def init_align()

if __name__ == '__main__':

    start = [5, 5]
    end = [10, 12]
    # wiringpi.wiringPiSetup()
    # wiringpi.pinMode(PWM1, 1)
    # wiringpi.pinMode(A1, 1)
    # wiringpi.pinMode(PWM2, 1)
    # wiringpi.pinMode(B1, 1)

    # forward(forward_dist)
    # time.sleep(3)
    # backward(backward_dist)
    # time.sleep(3)
    # left(left_dist)
    # time.sleep(3)
    # left(left_dist)
    # time.sleep(3)
    # left(left_dist)
    # time.sleep(3)
    # left(left_dist)
    # time.sleep(3)

    # time.sleep(20)
    init_direction = '+y'
    direction = init_direction

    listdata = []
    lastangle = 0
    #actor = PPO.actor_net(1)
    #actor.load_state_dict(torch.load('Actor_Model.pth')['net'])
    conn1, conn2 = multiprocessing.Pipe()
    P_child = multiprocessing.Process(target=get_raw_info, args=(conn2,))
    P_child.start()

    now = start

    grid = np.zeros([100,100])
    #fig1 = plt.figure()
    while True:
        align_counter = 0
        # print(get_state(direction, now, end, conn1))  # now, end: list
        if now == end:
            break
        else:
            grid = make_grid(10, now, grid, conn1)
            #raw_state, state = get_state(direction,now,end,conn1)
            #plt.pause(0.1)
            #fig1.clf()
            #time.sleep(10)
            # print(raw_state)
            # align(raw_state, conn1)
            # pi = actor.forward(state)
            # action = choose_action(pi)
            # if collision_detect(state, action):
            #     while collision_detect(state, action):
            #         action = choose_action(pi)

            #print(action)
            #print("||||||||||||||||")
            # action = 'None'
            #time.sleep(0.5)
            #next, direction = do_action(now, direction, action)
            # align_counter = 3#(align_counter + 1) % 4
            #now = next
            #time.sleep(0.5)
            # if align_counter == 3:
            # align(raw_state, conn1)
            # print("Align done!")
    P_child.join()