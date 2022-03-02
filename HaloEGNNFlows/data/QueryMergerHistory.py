import requests
import time
import os
import glob
import numpy as np
import multiprocessing

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"0412dcf89b4b7641266ac931701af034"}
def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    return r

# Query Simulation
r        = get(baseUrl)
sim_name = 'TNG300-1'
sim      = [sim['url'] for sim in r['simulations'] if sim['name'] == sim_name][0]
sim_query = get(sim)

# Get the last snapshot at z = 0
snaps = get(sim_query['snapshots'])
snap_z0 = get(snaps[-1]['url'])

# define the mass range of interest
h = 0.6774
mass_min = 0.5*10**12 / 1e10 * h
mass_max = 2.5*10**12 / 1e10 * h

# form the search_query string by hand for once
search_query = "?mass__gt=" + str(mass_min) + "&mass__lt=" + str(mass_max)

# Find all subhalos with mass between the range specified
url = snap_z0['subhalos'] + search_query
subs = get(url, {"limit":100000})



full_file_list = [f"sublink_{r['id']}.hdf5" for r in subs['results']]
existing_files = glob.glob('sublink*.hdf5')
remaining_files = sorted(list(set(full_file_list) - set(existing_files)))
assert len(full_file_list) - len(existing_files) == len(remaining_files)
remaining_urls = [r for r in subs['results'] if f"sublink_{r['id']}.hdf5" in remaining_files ]

print(len(remaining_files), len(remaining_urls))

def download_output_files(inputs):
  output_file, s = inputs
  sub = get(s['url'])
  if os.path.exists(output_file):
    return
  os.system(f'wget -O {output_file} --content-disposition {sub["trees"]["sublink"]} --header="API-Key: 0412dcf89b4b7641266ac931701af034"')
  print(f'{sub["trees"]["sublink"]} done!')
  return

pool_obj = multiprocessing.Pool(32)
pool_obj.map(download_output_files, zip(remaining_files,remaining_urls))
