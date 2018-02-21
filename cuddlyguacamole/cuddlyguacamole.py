import load

filename = 'sodium-chloride-example.npz'
start = time.time()

def optimize():
    print('optimizer')
    load_input_file()

def load_input_file():
    load.load_input_file(filename)
    print('loaded')