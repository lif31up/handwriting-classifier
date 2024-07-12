import argparse
def train(path: str, save_to: str, iters: int):
  import src.train as trainer
  trainer.main(path, save_to, iters)
# train

def eval(path: str):
  import src.eval as evaler
  evaler.main(path)
# eval

def main():
  # main cmd
  parser = argparse.ArgumentParser(description="cmd")
  parser.add_argument("--path", type=str, help="--path <path to your model>")

  # subcmd
  subparser = parser.add_subparsers(description="subcmd")
  ## train
  parser_train = subparser.add_parser("train")
  parser_train.add_argument("--path", type=str, help="--path <path to your train set>")
  parser_train.add_argument("--save-to", type=str, help="--save-to <path to save your weight>")
  parser_train.add_argument("--iters", type=int, help="--iters <num of iteration>")
  parser_train.set_defaults(func=lambda kwarg: train(kwarg.path, kwarg.save_to, kwarg.iters))

  args = parser.parse_args()
  if hasattr(args, 'func'): args.func(args)
  else: eval(args.path)
# main

if __name__ == "__main__": main()