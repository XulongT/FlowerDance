from args import parse_train_opt
from EDGE import EDGE


def inpaint(opt):
    checkpoint_path = './runs/train/uniform2/weights/train-3700.pt'
    model = EDGE(checkpoint_path=checkpoint_path)
    model.inpaint_loop(opt)

if __name__ == "__main__":
    opt = parse_train_opt()
    inpaint(opt)
