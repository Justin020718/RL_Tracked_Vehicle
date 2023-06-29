import multiprocessing as mp
impoe
def P1(shared_mem, mem_L):
    mem_L.acquire()
    for i in range(8640):
        shared_mem[i] = -1
    mem_L.release()

def main():
    shared_mem = mp.Manager().Array('f', range(4320 * 2))
    mem_L = mp.Lock()
    P_child = mp.Process(target=P1, args=(shared_mem, mem_L))
    P_child.start()
    time.sleep(4)
    mem_L.acquire()
    print(shared_mem)
    mem_L.release()
    P_child.join()
if __name__ == '__main__':
    main()