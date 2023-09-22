import wandb
api = wandb.Api()

projects = api.projects("adhikary-sandesh")
for p in projects:
    print(p)

domain = 'cartpole'
encoder_mode = 'dbc'

entity, project = "adhikary-sandesh", f"{domain}-invariance-exp-{encoder_mode}"
runs = api.runs(entity + "/" + project)

for run  in runs:
    print(run)