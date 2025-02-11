import argparse
import src.train as train
import src.eval as eval

def main():
  # main cmd
  parser = argparse.ArgumentParser(description="handwritten character recognition")
  parser.add_argument("--model", type=str, help="path to your model")
  parser.add_argument("--path", type=str, help="path to your test set")

  # subcmd
  subparser = parser.add_subparsers(description="subcommands", dest="subcmd")
  parser_train = subparser.add_parser("train")
  parser_train.add_argument("--path", type=str, help="path to your training set")
  parser_train.add_argument("--save-to", type=str, help="path to save the model")
  parser_train.add_argument("--iters", type=int, help="number of iterations")
  parser_train.set_defaults(func=lambda kwarg: train.main(path=kwarg.path, save_to=kwarg.save_to, iters=kwarg.iters))

  args = parser.parse_args()
  if hasattr(args, 'func'): args.func(args)
  else: eval.main(model=args.model, path=args.path)
# main

if __name__ == "__main__": main()