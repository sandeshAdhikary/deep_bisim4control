# import argparse
# import pickle
# from warnings import warn
# # Load all registered models and trainers
# # This is needed to load model/trainer classes from pickles
# from trainer.utils import import_registered_classes
# from src.evaluator import Evaluator
# # TODO: This needs to be registered
# from src.evaluator import BisimRLEvalautor
# import_registered_classes(globals())

# if __name__ == "__main__":

#     args = argparse.ArgumentParser()
#     args.add_argument("--eval_packet", type=str, required=True)
#     args.add_argument
#     args = args.parse_args()

#     with open(args.eval_packet, 'rb') as f:
#         eval_packet = pickle.load(f)

#     # Load evaluator
#     evaluator = eval_packet['evaluator_class'](eval_packet['evaluator_config'])
#     # Load model
#     model = eval_packet['model_class'](eval_packet['model_config'])
#     evaluator.set_model(model)
#     # Log and output files
#     eval_log_file = eval_packet.get('eval_log_file')
#     eval_output_file = eval_packet.get('eval_output_file')
#     evaluator.evaluate(eval_log_file=eval_log_file,
#                        eval_output_file=eval_output_file
#                        )
