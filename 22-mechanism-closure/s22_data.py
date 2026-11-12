"""Reuse audited read-only packet selection; block all target reads before sealing."""
from s22_core import HERE, access
import s15_data as previous
previous.HERE = HERE
previous.access = access
PROTOCOL = previous.PROTOCOL
shen_train = previous.shen_train
shen_cell = previous.shen_cell
roles = previous.roles
