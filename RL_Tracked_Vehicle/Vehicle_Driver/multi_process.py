#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import print_function
import serial
import torch
import PPO
import time
import wiringpi
import multiprocessing
import ctypes
import numpy as np
import random
import A_star
#o-------------x(right)
#|
#|
#|
#|
#|
#y(up)(init_direction)
PWM1 = 15
A1 = 14
PWM2 = 10
B1 = 9
raw_msg = []

forward_dist = 222
backward_dist = 222
left_dist = 180
right_dist = 175

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

def get_raw_info(conn, read_flag, L):
  ser = serial.Serial("/dev/ttyUSB0", 230400, timeout=5)
  if ser.is_open == False:
    ser.open()
  msg = []
  while True:
    data = ser.read(2)                         # 读取2个字节数据
    if data[0] == 0x54 and data[1] == 0x2C:    # 判断是否为数据帧头
      data = ser.read(45)                     # 是数据帧头就读取整一帧，去掉帧头之后为45个字节
      start_angle = ((data[3]*256+data[2])/100.0 -0.5) % 360
      end_angle = ((data[41]*256+data[40])/100.0 -0.5) % 360
      angle = start_angle
      if end_angle > start_angle:
        step = (end_angle - start_angle) / 11
      else:
        step = (end_angle + 360 -start_angle)/11
      for x in range(4, 40, 3):                         # 2个字节的距离数据，1个信号强度数据，步长为3
        msg.append([angle, data[x+1]*256+data[x]])
        angle = angle + step #角分辨率
    L.acquire()
    if read_flag == False:
      msg = []
    L.release()
    
    if len(msg) >= 4320:
      conn.send(msg)
      msg = []

def get_state(direction, conn):
    state = []
    threshold = 270
    raw_threshold = 180
    up_dist = []
    down_dist = []
    left_dist = []
    right_dist = []
    global raw_msg
    raw_msg = conn.recv()
    raw_msg = conn.recv()
    raw_msg = conn.recv()
    raw_msg = conn.recv()
    raw_msg = conn.recv()
    raw_msg = conn.recv()
    #print(raw_msg)
    #print(len(raw_msg))
    for angle, distance in raw_msg:
        if angle >=325 or angle <= 35:
            up_dist.append(distance)
        elif angle >= 55 and angle <= 125:
            left_dist.append(distance)
        elif angle >= 145 and angle <=215:
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
    return raw_state, state  
  
def forward(distance):
    #wiringpi.pwmSetRange(128)
    wiringpi.softPwmCreate(PWM1, 0, 100)
    wiringpi.softPwmCreate(PWM2, 0, 100)
    wiringpi.digitalWrite(A1, 1)
    wiringpi.digitalWrite(B1, 0)
    wiringpi.softPwmWrite(PWM1, 80)
    wiringpi.softPwmWrite(PWM2, 24)
    i = 0
    while True:
      i = i + 1
      if i == distance:
        wiringpi.softPwmWrite(PWM1, 100)
        wiringpi.softPwmWrite(PWM2, 0)
        return
      time.sleep(0.01)

def backward(distance):
    #wiringpi.pwmSetRange(128)
    wiringpi.softPwmCreate(PWM1, 0, 100)
    wiringpi.softPwmCreate(PWM2, 0, 100)
    wiringpi.digitalWrite(A1, 0)
    wiringpi.digitalWrite(B1, 1)
    wiringpi.softPwmWrite(PWM1, 20)
    wiringpi.softPwmWrite(PWM2, 80)
    i = 0
    while True:
      i = i + 1
      if i == distance:
        wiringpi.softPwmWrite(PWM2, 100)
        wiringpi.softPwmWrite(PWM1, 0)
        return
      time.sleep(0.01)

def left(time1, speed = 20):
    wiringpi.softPwmCreate(PWM2, 0, 100)
    wiringpi.softPwmCreate(PWM1, 0, 100)
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
  
def right(time1, speed = 20):
    wiringpi.softPwmCreate(PWM1, 0, 100)
    wiringpi.softPwmCreate(PWM2, 0, 100)
    wiringpi.digitalWrite(A1, 1)
    wiringpi.digitalWrite(B1, 1)
    wiringpi.softPwmWrite(PWM1, 100 - speed)
    wiringpi.softPwmWrite(PWM2, 100 - speed)
    i = 0
    while True:
      if i == time1:
        wiringpi.softPwmWrite(PWM2, 100)
        wiringpi.softPwmWrite(PWM1, 100)
        return
      i = i + 1
      time.sleep(0.01)

def align_up(conn):
    L, R = get_angle_distance(343, 17, conn)
    print("L ={}".format(L))
    print("R ={}".format(R))
    time.sleep(1)
    L1, R1 = get_angle_distance(343, 17, conn)
    while abs(L1-L) >= 2 or abs(R1-R) >= 2:
      time.sleep(0.5)
      L, R = get_angle_distance(343, 17, conn)
      time.sleep(0.5)
      L1, R1 = get_angle_distance(343, 17, conn)
    threshold = 5
    #return
    if (L - R) >= threshold or (R - L) >= threshold:
      while True:
        if L - R >= threshold:
          right(20, 5)
        elif (R - L) >= threshold:
          left(20, 5)

        time.sleep(0.5)
        L1, R1 = get_angle_distance(343, 17, conn)
        time.sleep(0.5)
        L2, R2 = get_angle_distance(343, 17, conn)
        while abs(L1-L2) >= 1 or abs(R1-R2) >= 1:
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
    while abs(L1-L) >= 2 or abs(R1-R) >= 2:
      time.sleep(0.5)
      L, R = get_angle_distance(197, 163, conn)
      time.sleep(0.5)
      L1, R1 = get_angle_distance(197, 163, conn)
    threshold = 5
    #return
    if (L - R) >= threshold or (R - L) >= threshold:
      while True:
        if L - R >= threshold:
          left(20, 5)
        elif (R - L) >= threshold:
          right(20, 5)

        time.sleep(0.5)
        L1, R1 = get_angle_distance(197, 163, conn)
        time.sleep(0.5)
        L2, R2 = get_angle_distance(197, 163, conn)
        while abs(L1-L2) >= 1 or abs(R1-R2) >= 1:
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
    while abs(L1-L) >= 2 or abs(R1-R) >= 2:
      time.sleep(0.5)
      L, R = get_angle_distance(287, 253, conn)
      time.sleep(0.5)
      L1, R1 = get_angle_distance(287, 253, conn)
    threshold = 5
    #return
    if (L - R) >= threshold or (R - L) >= threshold:
      while True:
        if L - R >= threshold:
          left(20, 5)
        elif (R - L) >= threshold:
          right(20, 5)

        time.sleep(0.5)
        L1, R1 = get_angle_distance(287, 253, conn)
        time.sleep(0.5)
        L2, R2 = get_angle_distance(287, 253, conn)
        while abs(L1-L2) >= 1 or abs(R1-R2) >= 1:
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
    while abs(L1-L) >= 2 or abs(R1-R) >= 2:
      time.sleep(0.5)
      L, R = get_angle_distance(73, 107, conn)
      time.sleep(0.5)
      L1, R1 = get_angle_distance(73, 107, conn)
    threshold = 5
    #return
    if (L - R) >= threshold or (R - L) >= threshold:
      while True:
        if L - R >= threshold:
          right(20, 5)
        elif (R - L) >= threshold:
          left(20, 5)

        time.sleep(0.5)
        L1, R1 = get_angle_distance(73, 107, conn)
        time.sleep(0.5)
        L2, R2 = get_angle_distance(73, 107, conn)
        while abs(L1-L2) >= 1 or abs(R1-R2) >= 1:
          time.sleep(0.5)
          L1, R1 = get_angle_distance(73, 107, conn)
          time.sleep(0.5)
          L2, R2 = get_angle_distance(73, 107, conn)
        print("L ={}".format(L1))
        print("R ={}".format(R1))
        if abs(L1 - R1) <= threshold:
          break

def align(raw_state, conn):
    #raw_state = get_raw_info(conn) # in the order of up, down, left, right
    if sum(raw_state) == 0:
      print("Didn't find any walls!")
      return # no wall to align, do nothing
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
    if p<=pi[0]:
        return 'up'
    elif p<=pi[0]+pi[1]:
        return 'down'
    elif p<=pi[0]+pi[1]+pi[2]:
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
      next = [x, y+1]
    elif action == 'down':
      next = [x, y-1]
    elif action == 'left':
      next = [x-1, y]
    else:
      next = [x+1, y]
    return next, next_direction

def collision_detect(state, action):
    if (state[0] == 1 and action == 'down') or (state[1] == 1 and action == 'up') or (state[2] == 1 and action == 'left') or (state[3] == 1 and action == 'right'):
      return True
    else:
      return False

def is_neighbor(a, b):
  # a in pos now, b is from node list, return action(a->b)
  x_a, y_a = a
  x_b, y_b = b
  if x_a == x_b and y_a - y_b == 1:
    return 'down'
  elif x_a == x_b and y_b - y_a == 1:
    return 'up'
  elif y_a == y_b and x_a - x_b == 1:
    return 'left'
  elif y_a == y_b and x_b - x_a == 1:
    return 'right'
  else:
    return None

def get_action_from_node(start, path):
  action_list = []
  now =start
  while path != []:
    for node in path:
      action = is_neighbor(now, node)
      if action is not None:
        action_list.append(action)
        now = node
        path.remove(node)
        break
  return action_list

#def init_align()

if __name__ == '__main__':
    
    start = [1, 2]
    end = [9, 1]
    mode = 1 # mode 0 will make the map, while mode 1 uses the map
    
    wiringpi.wiringPiSetup()
    wiringpi.pinMode(PWM1,1)
    wiringpi.pinMode(A1,1)
    wiringpi.pinMode(PWM2,1)
    wiringpi.pinMode(B1,1)

    init_direction = '+y'
    direction = init_direction

    listdata = []
    lastangle = 0
    actor = PPO.actor_net(1)
    actor.load_state_dict(torch.load('Actor_Model.pth')['net'])
    conn1, conn2 = multiprocessing.Pipe()
    L = multiprocessing.Lock()
    read_flag = multiprocessing.Manager().Value(ctypes.c_bool, True)
    P_child = multiprocessing.Process(target = get_raw_info, args=(conn2, read_flag, L))
    P_child.start()
    
    # while True:
    #   L, R = get_angle_distance(340, 20, conn1)
    #   print(L)
    #   print(R)
    now = start
    #time.sleep(30)
    prev_raw = [6,6,6,6]
    prev_action = None
    # forward(forward_dist)
    # time.sleep(5)
    # forward(forward_dist)
    # time.sleep(1)
    # left(left_dist)
    # time.sleep(1)
    # right(right_dist)
    # time.sleep(1)
    if mode == 0:
      map = np.ones(tuple([100,100]))
      map[now] = 0
      map[end] = 0
      while True:
        align_counter = 0
        #print(get_state(direction, now, end, conn1))  # now, end: list
        if now == end:
          # save map to disk
          np.save('saved_map.npy', np.array(map))
          break
        else:
          time.sleep(0.5)
          raw_state, state = get_state(direction, conn1)
          if raw_state == prev_raw:
            flag = True
          else:
            flag = False
          prev_raw = raw_state
          state = state + now
          state = state + end
          
          # build map
          x, y = now
          map[now] = 0
          if state[0] == 0:
            map[x,y-1] = 0
          if state[1] == 0:
            map[x,y+1] = 0
          if state[2] == 0:
            map[x-1, y] = 0
          if state[3] == 0:
            map[x+1, y] = 0

          pi = actor.forward(torch.FloatTensor(state))
          if flag == False:
            action = choose_action(pi)
            if collision_detect(state, action):
              while collision_detect(state, action):
                action = choose_action(pi)
          else:
            action = prev_action

          # None action is for test
          #action = 'None'
          print(state)
          print(action)
          a = input()
          if a == 'yes':
            L.acquire()
            read_flag = False
            L.release()
            next, direction = do_action(now, direction, action)
            prev_action = action
            align_counter = (align_counter + 1) % 4
          else:
            next, direction = do_action(now, direction, 'None')
            prev_action = choose_action(pi)
          
          read_flag = True
          now = next
          if align_counter == 3:
            align(raw_state, conn1)
            #print("Align done!")
      P_child.join()

    elif mode ==  1:
      #time.sleep(30)
      map = np.load('saved_map.npy')
      flag, path = A_star.solve(map.tolist(),start, end)
      action_list = get_action_from_node(start, path)
      print(action_list)
      time.sleep(10)
      for action in action_list:
        next, direction = do_action(now, direction, action)
        now = next
        #time.sleep(0.5)
        raw_state, state = get_state(direction, conn1)
        align(raw_state, conn1)
      P_child.join()
        