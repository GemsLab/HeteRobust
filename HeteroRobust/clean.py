import signac
import fire
from pathlib import Path
import os

def _do_remove(*jobID):
    input_str = input("Proceed removing? [Y/n]")
    if input_str.lower() in ["y", "yes"]:
        for job in jobID:
            print(f"Removing job {job.id}")
            job.remove()
    else:
        print("Abort.")

def unsuccess(*jobID):
    '''
    Remove all signac jobs which is not marked as succeeded in their job document.
    If jobID is provided, removal only takes place on the provided jobs.
    In command line, this option can be use together with signac command, e.g.
    ```
    python -m HeteroRobust.clean unsuccess $(signac find type test)
    ```
    '''
    if len(jobID) > 0:
        targetJobSet = set(jobID)
    removeJobList = []
    for job in project.find_jobs():
        if len(jobID) > 0 and job.id not in targetJobSet:
            continue
        if not job.doc.get("success", False):
            print(f"Found unsuccessful {job.sp.get('type', 'unknown type')} job: {job.id}")
            removeJobList.append(job)
    _do_remove(*removeJobList)

def success(use_runner=False):
    cwd = Path(os.getcwd())
    pathList = []
    sp_filter = dict()
    if use_runner:
        sp_filter = dict(use_runner=True)
    for job in project.find_jobs(
        sp_filter,
        dict(success=True)
    ):
        pathList.append(f'{str(Path(job.workspace()).relative_to(cwd))}')
    print(' '.join(pathList))

def remove(*jobID):
    '''
    Unconditionally remove jobs specified by the jobID.
    '''
    print(f"About to remove jobs: {jobID}")
    removeJobList = [project.open_job(id=i) for i in jobID]
    _do_remove(*removeJobList)

if __name__ == "__main__":
    project = signac.get_project()
    fire.Fire()
