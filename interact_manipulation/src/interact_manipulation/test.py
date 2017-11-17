class CostWithTime():
    def __init__(self, t, get_cost_func):
        self.time = t
        self.get_cost_func = get_cost_func
        print("init time {}".format(self.time))
    def __call__(self, config):        
        return self.get_cost_func(config, self.time)

def get_cost_func(t):
    print("setting up cost func t= {}".format(t))
    return CostWithTime(t, get_cost_with_t)

def get_cost_with_t(t):
    print("doing something with t: {}".format(t))

def call_in_order(funcs):
    config = 1
    for func in funcs:
        func(config)

def main():
    funcs = []
    for t in range(3):
        funcs.append(get_cost_func(t))
    call_in_order(funcs)

if __name__ == '__main__':
    main()