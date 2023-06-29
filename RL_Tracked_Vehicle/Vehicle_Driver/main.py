#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import print_function
import serial
import torch
import PPO
import time
import wiringpi
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

def get_angle_distance(angle):
  ser = serial.Serial("/dev/ttyUSB0", 230400, timeout=5)
  if ser.is_open == False:
    ser.open()
  angle_dist = []
  while angle_dist == []:
    data = ser.read(2)
    if data[0] == 0x54 and data[1] == 0x2C:
      data = ser.read(45)
      start_angle = (data[3]*256+data[2])/100.0
      #print(start_angle)
      for x in range(4, 40, 3):
        if start_angle >= ((angle-4)%360) and start_angle <= ((angle-2)%360):
          angle_dist.append(data[x+1]*256+data[x])
  #print(angle_dist)
  ser.close()
  return (sum(angle_dist) / len(angle_dist))
  

def get_raw_state():
  ser = serial.Serial("/dev/ttyUSB0", 230400, timeout=5)
  if ser.is_open == False:
    ser.open()
  threshold = 150
  state = []
  up_dist = []
  down_dist = []
  left_dist = []
  right_dist = []
  for i in range(100):
    data = ser.read(2)                         # 读取2个字节数据
    if data[0] == 0x54 and data[1] == 0x2C:    # 判断是否为数据帧头
      data = ser.read(45)                     # 是数据帧头就读取整一帧，去掉帧头之后为45个字节
      start_angle = (data[3]*256+data[2])/100.0
      for x in range(4, 40, 3):                         # 2个字节的距离数据，1个信号强度数据，步长为3
        if start_angle >325 or start_angle <= 29:
          up_dist.append(data[x+1]*256+data[x])
        elif start_angle > 55 and start_angle <= 119:
          right_dist.append(data[x+1]*256+data[x])
        elif start_angle > 145 and start_angle <=209:
          down_dist.append(data[x+1]*256+data[x])
        elif start_angle > 235 and start_angle <= 299:
          left_dist.append(data[x+1]*256+data[x])
  
  up_avg = sum(up_dist) / len(up_dist)
  down_avg = sum(down_dist) / len(down_dist)
  left_avg = sum(left_dist) / len(left_dist)
  right_avg = sum(right_dist) / len(right_dist)
  if up_avg >= threshold:
    state.append(0)
  else:
    state.append(1)
  if down_avg >= threshold:
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
  ser.close()
  return state # up, down, left, right

def get_state(direction, now, end):
  ser = serial.Serial("/dev/ttyUSB0", 230400, timeout=5)       # window系统，需要先通过设备管理器确认串口COM号
  if ser.is_open == False:
    ser.open()
  
  threshold = 150
  state = []
  up_dist = []
  down_dist = []
  left_dist = []
  right_dist = []
  for i in range(100):
    data = ser.read(2)                         # 读取2个字节数据
    if data[0] == 0x54 and data[1] == 0x2C:    # 判断是否为数据帧头
      data = ser.read(45)                     # 是数据帧头就读取整一帧，去掉帧头之后为45个字节

      # for x in range(44):                   # 打印原始数据，默认不打印
      #  print('%#.2x '%ord(data[x]),end="\t")
      # print("\n")

      #listdata.insert(0, "转速（度/秒）:")
      #listdata.insert(1, data[1]*256+data[0])           # 转速：高字节在后，低字节在前，复合后再转换成十进制

      #listdata.insert(2, "起始角度（度）:")
      start_angle = (data[3]*256+data[2])/100.0
      #print(start_angle)
      #listdata.insert(3, (data[3]*256+data[2])/100.0)   # 原始角度为方便传输放大了100倍，这里要除回去
      #listdata.insert(4, "距离（mm）|光强 *12个点 :")
      for x in range(4, 40, 3):                         # 2个字节的距离数据，1个信号强度数据，步长为3
        if start_angle >325 or start_angle <= 29:
          up_dist.append(data[x+1]*256+data[x])
        elif start_angle > 55 and start_angle <= 119:
          left_dist.append(data[x+1]*256+data[x])
        elif start_angle > 145 and start_angle <=209:
          down_dist.append(data[x+1]*256+data[x])
        elif start_angle > 235 and start_angle <= 299:
          right_dist.append(data[x+1]*256+data[x])
  up_avg = sum(up_dist) / len(up_dist)
  down_avg = sum(down_dist) / len(down_dist)
  left_avg = sum(left_dist) / len(left_dist)
  right_avg = sum(right_dist) / len(right_dist)

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
  ser.close()
  return torch.FloatTensor(state)  
  
def forward(distance):
    #wiringpi.pwmSetRange(128)
    wiringpi.softPwmCreate(PWM1, 0, 25)
    wiringpi.softPwmCreate(PWM2, 0, 25)
    wiringpi.digitalWrite(A1, 1)
    wiringpi.digitalWrite(B1, 0)
    wiringpi.softPwmWrite(PWM1, 20)
    wiringpi.softPwmWrite(PWM2, 5)
    i = 0
    while True:
      i = i + 1
      if i == distance:
        wiringpi.softPwmWrite(PWM1, 25)
        wiringpi.softPwmWrite(PWM2, 0)
        return
      time.sleep(0.1)

def backward(distance):
    #wiringpi.pwmSetRange(128)
    wiringpi.softPwmCreate(PWM1, 0, 25)
    wiringpi.softPwmCreate(PWM2, 0, 25)
    wiringpi.digitalWrite(A1, 0)
    wiringpi.digitalWrite(B1, 1)
    wiringpi.softPwmWrite(PWM1, 5)
    wiringpi.softPwmWrite(PWM2, 20)
    i = 0
    while True:
      i = i + 1
      if i == distance:
        wiringpi.softPwmWrite(PWM2, 25)
        wiringpi.softPwmWrite(PWM1, 0)
        return
      time.sleep(0.1)

def left(distance):
    wiringpi.softPwmCreate(PWM1, 0, 25)
    wiringpi.softPwmCreate(PWM2, 0, 25)
    wiringpi.digitalWrite(A1, 0)
    wiringpi.digitalWrite(B1, 0)
    wiringpi.softPwmWrite(PWM1, 5)
    wiringpi.softPwmWrite(PWM2, 5)
    i = 0
    while True:
      if i == distance:
        wiringpi.softPwmWrite(PWM2, 0)
        wiringpi.softPwmWrite(PWM1, 0)
        return
      i = i + 1
      time.sleep(0.01)
  
def right(distance):
    wiringpi.softPwmCreate(PWM1, 0, 25)
    wiringpi.softPwmCreate(PWM2, 0, 25)
    wiringpi.digitalWrite(A1, 1)
    wiringpi.digitalWrite(B1, 1)
    wiringpi.softPwmWrite(PWM1, 20)
    wiringpi.softPwmWrite(PWM2, 20)
    i = 0
    while True:
      if i == distance:
        wiringpi.softPwmWrite(PWM2, 25)
        wiringpi.softPwmWrite(PWM1, 25)
        return
      i = i + 1
      time.sleep(0.01)

def align_up():
    L = get_angle_distance(330)
    R = get_angle_distance(30)
    print("L ={}".format(L))
    print("R ={}".format(R))


def align(raw_state):
    raw_state = get_raw_state() # in the order of up, down, left, right
    if sum(raw_state) == 0:
      return # no wall to align, do nothing
    else:
      if raw_state[0] == 1:
        align_up()
      elif raw_state[1] == 1:
        align_down()
      elif raw_state[2] == 1:
        align_left()
      else:
        align_right()


if __name__ == '__main__':
    now = [1, 1]
    end = [12, 12]
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
    #forward(30)
    #print('foward done!')
    #time.sleep(3)
    #backward(30)
    #print('backward done!')
    #time.sleep(3) #left 131
    #right(131)
    #print('turn left done!')
    #time.sleep(3)
    #right(131)
    #time.sleep(3)
    #right(131)
    #time.sleep(3)
    #right(131)
    #time.sleep(3)
    #print('turn right done!')
    # LD14P波特率：230400     LD14波特率：115200
    # ser= serial.Serial('/dev/wheeltec_lidar', 115200)  # ubuntu，如果未修改串口别名，可通过 ll /dev 查看雷达具体端口再进行更改
    
    while True:
      time.sleep(1)
      align_up()
      #print(get_state(direction, now, end))  # now, end: list