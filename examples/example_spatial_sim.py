import example_spatial_utils as sutils
import argparse

parser = argparse.ArgumentParser()

iow_path = '../../id_spatial_sim/scenarios/covid_leicestershire/output_all/output_iow/networks'
parser.add_argument('--network_path',default=iow_path)
parser.add_argument('--prefix',default='leicestershire')
parser.add_argument('--n',default=128)

(args,args_extra) = parser.parse_known_args()
params_extra = {a.split('--')[1]:b for a,b in zip(args_extra[::2],args_extra[1::2])}

# Create model
m = sutils.load_network_model(iow_path,prefix=args.prefix,params_extra=params_extra)

# Run
for t in range(args.n):
    m.one_time_step()

# Print results
print(m.one_time_step_results())