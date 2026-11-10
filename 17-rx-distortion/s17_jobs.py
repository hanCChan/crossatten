from s17_core import HERE

def split_seed(j):return j.get('split_seed',j['seed'])
def cache_key(j):return f"{j['dataset']}_f{j['fold']}_{j['group']}_s{split_seed(j)}"
def run_id(j):return f"{j['phase']}__{j['dataset']}__{j['arm']}__f{j['fold']}__{j['group']}__s{j['seed']}"
def profiles_path(j):return HERE/'profiles'/f'{cache_key(j)}.json'
