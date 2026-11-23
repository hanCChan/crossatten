"""Fail closed: current user attachment explicitly reserves new-fit authorization."""
from s28_core import read,HERE,validate

if __name__=='__main__':
    registration=validate()
    raise SystemExit('NO TRAINING STARTED. Current scientific fit budget is 0. Review PLAN_A.json and the unchanged PLAN_C.original.json; a separate explicit user authorization and committed activation are required.')
