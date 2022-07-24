import signac
import multiprocessing
import argparse
import pandas
import json
import string
import itertools
import asyncio
import traceback
import numpy as np
from pathlib import Path
import sys
import os
import signal
import time
import socket
import datetime
from .modules.utils import atomic_overwrite, write_csv, list_value_format, dict_value_format
import logging
import io
import deeprobust
import re
from collections import deque
import tempfile
logging.basicConfig(format='[%(levelname)s %(name)s %(asctime)s] %(message)s', datefmt='%m/%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

slack_webhook_url = ""

formatter = string.Formatter()
templates_dict = {
    "NettackSession": {
        "evasion": "templates/nettack-evasion.txt",
        "poison": "templates/nettack-poison.txt",
        "poison_only": "templates/nettack-poison-only.txt",
        "perturb": "templates/nettack-perturb.txt"
    },
    "MetattackSession": {
        "evasion": "templates/metattack-evasion.txt",
        "poison": "templates/metattack-poison.txt",
        "poison_only": "templates/metattack-poison-only.txt",
        "clean_only": "templates/metattack-clean-only.txt",
        "perturb": "templates/metattack-perturb.txt"
    },
    "SparseSmoothingSession": {
        "clean_only": "templates/sparse-smoothing-cert.txt"
    }
}
template_path = Path(__file__).parent
cmd_template_cache = dict()
allowed_attack_phases = set(["poison", "evasion", "poison+evasion", "clean-only", "perturb"])
DISABLED_JOB_STR = "N/A"
RERUN_JOB_STR = "RERUN"
HOSTNAME = socket.gethostname()
dry_run_mode = False
debug_mode = False
detect_mode = False
proc_set = set()
project = signac.get_project()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--parallel_num", "-p", default=4, type=int)
    parser.add_argument("--dry_run", "-n", action="store_true")
    parser.add_argument("--get_status", "-s", action="store_true")
    parser.add_argument("--result_path",
                        default="{result_folder}/{config_name}.csv")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vsdebug", action="store_true")
    parser.add_argument("--notify", choices=[], default=[], nargs="*")
    parser.add_argument("--wait_pid", nargs="+", default=[], type=int)
    parser.add_argument("--reverse_order", action="store_true")
    parser.add_argument("--auto_retry", default=True, type=bool)
    args = parser.parse_args()
    return args

def build_from_json(json_path):
    with open(json_path, "r") as json_in:
        config_dict = json.load(json_in)
    vars_dict = config_dict["Vars"]
    for key, value in vars_dict.items():
        if type(value) is not list:
            vars_dict[key] = [value]
    # print(vars_dict)

    if "PerturbJobFilters" in config_dict:
        # Build the table for all perturb jobs to consider
        perturb_job_filters_vars = set()
        for field in ["sp", "doc"]:
            for _, value in config_dict["PerturbJobFilters"][field].items():
                for _, var_name, _, _ in formatter.parse(value):
                    if var_name is None:
                        continue
                    elif var_name not in vars_dict:
                        raise ValueError(f"Undefined value {var_name} appeared in PerturbJobFilters!")
                    perturb_job_filters_vars.add(var_name)
        perturb_job_filters_vars = list(perturb_job_filters_vars)

        perturbJobDf = pandas.DataFrame(columns=perturb_job_filters_vars + ["perturbJobID"])
        for values in itertools.product(*[vars_dict[key] for key in perturb_job_filters_vars]):
            value_mapping = dict(zip(perturb_job_filters_vars, values))
            formatted_job_filters = dict_value_format(config_dict["PerturbJobFilters"], value_mapping)
            for job in project.find_jobs(
                filter=formatted_job_filters["sp"], doc_filter=formatted_job_filters["doc"]
            ):
                data = value_mapping.copy()
                data["perturbJobID"] = job.id
                perturbJobDf.loc[len(perturbJobDf)] = pandas.Series(data)

    elif "datasetName" in config_dict:
        # Build the table for all dataset names to consider
        datasetName_vars = set()
        for value in config_dict["datasetName"]:
            for _, var_name, _, _ in formatter.parse(value):
                if var_name is None:
                    continue
                elif var_name not in vars_dict:
                    raise ValueError(f"Undefined value {var_name} appeared in PerturbJobFilters!")
                datasetName_vars.add(var_name)
        datasetName_vars = list(datasetName_vars)
        perturbJobDf = pandas.DataFrame(columns=datasetName_vars + ["datasetName"])

        for arg in config_dict["datasetName"]:
            datasetName_set = set()
            # Scan variables
            for _, var_name, _, _ in formatter.parse(arg):
                if var_name is None:
                    continue
                elif var_name not in vars_dict:
                    raise ValueError(f"Undefined value {var_name} appeared in model arguments!")
                datasetName_set.add(var_name)
            
            datasetName_set = list(datasetName_set)
            # Format variables
            if len(datasetName_set) > 0:
                for values in itertools.product(*[vars_dict[key] for key in datasetName_set]):
                    data = dict(zip(datasetName_set, values))
                    formatted_datasetName = arg.strip().format(**data)
                    data["datasetName"] = formatted_datasetName

                    # For synthetic graphs, check if the corresponding files exist
                    re_necessity = re.match("necessity-cora:(.+)", formatted_datasetName)
                    re_syn = re.match("syn-cora:(.+)", formatted_datasetName)
                    if re_necessity:
                        filename = re_necessity.group(1) + ".npz"
                        if (Path("datasets") / "necessity-cora" / filename).exists():
                            perturbJobDf.loc[len(perturbJobDf)] = pandas.Series(data)
                    elif re_syn:
                        filename = re_syn.group(1) + ".npz"
                        if (Path("datasets") / "syn-cora" / filename).exists():
                            perturbJobDf.loc[len(perturbJobDf)] = pandas.Series(data)
                    else:
                        perturbJobDf.loc[len(perturbJobDf)] = pandas.Series(data)
                    
            else: # When there are no variables in model args for a model
                data = {
                    "datasetName": arg
                }
                perturbJobDf.loc[len(perturbJobDf)] = pandas.Series(data)
        
    if "randomSeed" in config_dict:
        perturbJobDf = perturbJobDf.merge(
            pandas.Series(config_dict["randomSeed"], name="randomSeed"), how="cross")

    if "SessionConfig" in config_dict:
        if not debug_mode:
            sessionConfigList = config_dict["SessionConfig"]
        else:
            sessionConfigList = [value + " --debug" for value in config_dict["SessionConfig"]]

        perturbJobDf = perturbJobDf.merge(
            pandas.Series(sessionConfigList, name="SessionConfig"), how="cross")
    
    if "TemplateVars" in config_dict:
        for key, value in config_dict["TemplateVars"].items():
            perturbJobDf = perturbJobDf.merge(
                pandas.Series(value, name=key), how="cross")

    # if "SessionConfig" in config_dict:
    #     session_config_str_list = []
    #     varnames = config_dict["SessionConfig"].keys()
    #     for combinations in itertools.product(*config_dict["SessionConfig"].values()):
    #         s_config_list = []
    #         for key, value in zip(varnames, combinations):
    #             s_config_list += [f"--{key}", f"{value}"]
    #         session_config_str = ' '.join(s_config_list)
    #         session_config_str_list.append(session_config_str)
    #     perturbJobDf = perturbJobDf.merge(
    #         pandas.Series(session_config_str_list, name="SessionConfig"), how="cross")
    #     breakpoint()

    # Random shuffle perturbJobDf
    perturbJobDf = perturbJobDf.sample(frac=1, random_state=0)

    # Build the table for all experiment parameters
    model_arg_vars_set = set()
    for model_name, arg_list in config_dict["model"].items():
        for value in arg_list:
            for _, var_name, _, _ in formatter.parse(value):
                if var_name is None:
                    continue
                elif var_name not in vars_dict:
                    raise ValueError(f"Undefined value {var_name} appeared in model arguments!")
                model_arg_vars_set.add(var_name)
    model_arg_vars = list(model_arg_vars_set)
    modelArgDf = pandas.DataFrame(columns=model_arg_vars + ["model", "model_arg", "Attack Phase"])

    for model_name, arg_list in config_dict["model"].items():
        for arg in arg_list:
            arg = arg.rsplit("#", 1)[0].strip() # Allow comments at the end of string

            model_arg_set = set()
            # Scan variables
            for _, var_name, _, _ in formatter.parse(arg):
                if var_name is None:
                    continue
                elif var_name not in vars_dict:
                    raise ValueError(f"Undefined value {var_name} appeared in model arguments!")
                model_arg_set.add(var_name)
            
            model_arg_set = list(model_arg_set)
            # Format variables
            if len(model_arg_set) > 0:
                for values in itertools.product(*[vars_dict[key] for key in model_arg_set]):
                    data = dict(zip(model_arg_set, values))
                    data["model"] = model_name
                    formatted_arg = arg.strip().format(**data)
                    split_args = formatted_arg.split(":", 1)
                    data["model_arg"] = " ".join(split_args[-1].strip().split())
                    data["Attack Phase"] = split_args[0].strip()
                    assert data["Attack Phase"] in allowed_attack_phases, f"Invalid attack phase {data['Attack Phase']}: only {allowed_attack_phases} is allowed!"
                    modelArgDf.loc[len(modelArgDf)] = pandas.Series(data)

            else: # When there are no variables in model args for a model
                data = {
                    "model": model_name, 
                    "model_arg": " ".join(arg.split(":", 1)[-1].strip().split()),
                    "Attack Phase": arg.split(":", 1)[0].strip()
                }
                assert data["Attack Phase"] in allowed_attack_phases, f"Invalid attack phase {data['Attack Phase']}: only {allowed_attack_phases} is allowed!"
                modelArgDf.loc[len(modelArgDf)] = pandas.Series(data)
    
    # Random shuffle modelArgDf
    modelArgDf = modelArgDf.sample(frac=1, random_state=0)

    # Join two table using a simple cartesian product (require pandas >= 1.2.0)
    if len(set(perturbJobDf.columns) & set(modelArgDf.columns)) > 0:
        # TODO: handle if there are shared variables between perturbJobDf and modelArgDf
        raise NotImplementedError()
    else:
        expConfigDf = perturbJobDf.merge(modelArgDf, how='cross')

    for key, value in vars_dict.items():
        if key not in expConfigDf.columns and len(value) == 1:
            expConfigDf[key] = value[0]

    # Initialize columns for job ID store
    expConfigDf["AttackSession"] = config_dict["AttackSession"]
    expConfigDf["cleanJobID"] = None
    expConfigDf["evasionJobID"] = None
    expConfigDf["poisonJobID"] = None
    if len(config_dict["Sorts"]) > 0:
        expConfigDf.sort_values(config_dict["Sorts"], inplace=True)
        expConfigDf.reset_index(drop=True, inplace=True)
    expConfigDf.loc[expConfigDf["Attack Phase"].isin(["evasion", "clean-only", "perturb"]), "poisonJobID"] = DISABLED_JOB_STR
    expConfigDf.loc[expConfigDf["Attack Phase"].isin(["poison", "clean-only", "perturb"]), "evasionJobID"] = DISABLED_JOB_STR
    
    perturbMask = (expConfigDf["Attack Phase"] == "perturb")
    if perturbMask.any():
        if "perturbJobID" not in expConfigDf.columns:
            expConfigDf["perturbJobID"] = DISABLED_JOB_STR
        expConfigDf.loc[perturbMask, "cleanJobID"] = DISABLED_JOB_STR
        expConfigDf.loc[perturbMask, "perturbJobID"] = None


    return expConfigDf

def create_digest(task_queue: asyncio.Queue, fail_queue: asyncio.Queue, 
                  message_queue: asyncio.Queue(), worker_list=None, expConfigDf=None,
                  is_running=None, add_csv=False):
    digest = ""
    total_task = None
    if expConfigDf is not None:
        total_task = len(expConfigDf)
        task_completed = total_task - task_queue.qsize() - fail_queue.qsize()
        if is_running is not None:
            task_completed -= sum(is_running)
        digest += f"{task_completed} of {total_task} jobs completed "
        digest += f"({task_completed / total_task:.1%}); "

    if is_running is not None:
        digest += f"{sum(is_running)} jobs running"
        if total_task:
            digest += f" ({sum(is_running) / total_task:.1%})"
        digest += "; "

    # Remaining and fail task info
    for key, value in dict(remaining=task_queue.qsize(), failed=fail_queue.qsize()).items():
        digest += f"{value} tasks {key}"
        if total_task:
            digest += f" ({value / total_task:.1%})"
        if key == "remaining":
            digest += "; "
    
    # Worker info
    if worker_list:
        alive_worker = sum([not worker.done() for worker in worker_list])
        digest += f"; {alive_worker} of {len(worker_list)} workers are still alive."

    if expConfigDf is not None and add_csv:
        digest += f"\n```\n{expConfigDf.to_csv()}\n```"

    return digest

async def slack_client(args, message_queue: asyncio.Queue()):
    error_interval = 600
    client_disabled = False
    if dry_run_mode or debug_mode or detect_mode:
        client_disabled = True
    logger = logging.getLogger("slack-client")
    try:
        from slack_sdk.webhook.async_client import AsyncWebhookClient
        import aiohttp
        webhook = AsyncWebhookClient(slack_webhook_url)
    except Exception as e:
        logger.error(f"Cannot start slack client: {e}")
        client_disabled = True

    pending_error_msg = []
    last_error = None
    task_list = [asyncio.create_task(message_queue.get())]
    delay_timer = False
    while True:
        await asyncio.wait(task_list, return_when=asyncio.FIRST_COMPLETED)
        if task_list[0].done():
            msg_level, msg = task_list[0].result()
            if len(msg) > 10000: 
                msg = msg[-10000:]
            if not client_disabled:
                if msg_level == logging.INFO:
                    response = await webhook.send(
                        text=f":information_source: INFO - HeteroRobust runner on `{HOSTNAME}` with config `{args.config}`:\n{msg}")
                    if response.status_code != 200 or response.body != "ok":
                        logger.error(f"Failed to send message to slack: {response}")
                elif msg_level == logging.ERROR:
                    if last_error is None or (datetime.datetime.now() - last_error).seconds >= error_interval:
                        response = await webhook.send(
                            text=f':x: ERROR - HeteroRobust runner on `{HOSTNAME}` with config `{args.config}`:\n{msg}')
                        if response.status_code != 200 or response.body != "ok":
                            logger.error(f"Failed to send message to slack: {response.status_code} - {response.body}")
                        last_error = datetime.datetime.now()
                    else:
                        if delay_timer == False:
                            delay_timer = True
                            remaining_sec = error_interval - (datetime.datetime.now() - last_error).seconds
                            task_list.append(asyncio.create_task(asyncio.sleep(remaining_sec)))
                        pending_error_msg.append(msg.replace("```", ""))
                elif msg_level == "completed":
                    for user in args.notify:
                        response = await webhook.send_dict(dict(
                            text=f':white_check_mark: Experiment completed - HeteroRobust runner on `{HOSTNAME}` with config `{args.config}`:\n{msg}',
                            channel=user
                        ))
                        if response.status_code != 200 or response.body != "ok":
                            logger.error(f"Failed to send message to slack: {response}")
                elif msg_level == "direct-info":
                    for user in args.notify:
                        response = await webhook.send_dict(dict(
                            text=f":information_source: INFO - HeteroRobust runner on `{HOSTNAME}` with config `{args.config}`:\n{msg}",
                            channel=user
                        ))
                        if response.status_code != 200 or response.body != "ok":
                            logger.error(f"Failed to send message to slack: {response}")
            message_queue.task_done()
            task_list[0] = asyncio.create_task(message_queue.get())
        
        if not client_disabled and len(task_list) > 1 and task_list[1].done():
            summary_msg = f":x: ERROR - HeteroRobust runner on `{HOSTNAME}` with config `{args.config}`: "
            summary_msg += f"{len(pending_error_msg)} error messages received in last {error_interval / 60:.1f} minutes. \n```\n"
            # for ind, msg in enumerate(pending_error_msg):
            #     summary_msg += f"\n====== Message {ind} ======\n"
            #     summary_msg += msg
            #     summary_msg += f"\n===========================\n"
            # summary_msg += "```\n"
            response = await webhook.send(text=summary_msg)
            if response.status_code != 200 or response.body != "ok":
                logger.error(f"Failed to send message to slack: {response}")
            last_error = datetime.datetime.now()
            pending_error_msg.clear()
            del task_list[1]
            delay_timer = False


async def progress_monitor(task_queue: asyncio.Queue, fail_queue: asyncio.Queue, 
                           message_queue: asyncio.Queue(), worker_list=None, 
                           expConfigDf=None, is_running=None, every_term=300, every_slack=3600):
    digest_logger = logging.getLogger("Digest")
    digest_logger.setLevel(logging.INFO)

    term_timer = asyncio.create_task(asyncio.sleep(every_term))
    slack_timer = asyncio.create_task(asyncio.sleep(every_slack))
    try:
        while True:
            await asyncio.wait([term_timer, slack_timer], return_when=asyncio.FIRST_COMPLETED)
            if term_timer.done():
                digest = create_digest(
                    task_queue, fail_queue, message_queue, worker_list, expConfigDf, is_running)
                digest_logger.info(f"{digest}")
                # Create new terminal timer
                term_timer = asyncio.create_task(asyncio.sleep(every_term))
            
            if slack_timer.done():
                digest = create_digest(
                    task_queue, fail_queue, message_queue, worker_list, expConfigDf,
                    is_running, add_csv=False)
                asyncio.create_task(message_queue.put((logging.INFO, digest)))
                # Create new slack timer
                slack_timer = asyncio.create_task(asyncio.sleep(every_slack))
            
    except asyncio.CancelledError:
        term_timer.cancel()
        slack_timer.cancel()

async def run_subprocess(attack_session, template_name, task_data, logger, message_queue, force_rerun=False):
    if dry_run_mode:
        return f"{template_name} results"
    else:
        template_key = f"{attack_session}/{template_name}"
        if template_key not in cmd_template_cache:
            template_file = templates_dict[attack_session][template_name]
            command_parts = []
            with open(str(template_path / template_file), "r") as f:
                for line in f:
                    command_parts.append(line.strip())
            command_template = ' '.join(command_parts)
            cmd_template_cache[template_key] = command_template
        else:
            command_template = cmd_template_cache[template_key]
        command = command_template.format(**task_data)
        existingJobList = list(project.find_jobs(doc_filter=dict(comment=command, success=True)))
        if len(existingJobList) > 2:
            raise RuntimeError(f"Found more than 2 existing jobs for the following command:\n{command}")
        elif len(existingJobList) == 1:
            logger.warning(f"Job already succeeded in {existingJobList[0].id}: {command}")
            return existingJobList[0].id
        elif not detect_mode:
            command_list = command.split()
            command_list = [sys.executable, "-u", *command_list[1:-1], "add_comment", command, "-", command_list[-1]]
            
            # Run the process
            if not debug_mode:
                proc = await asyncio.create_subprocess_exec(command_list[0], *command_list[1:], 
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                proc_set.add(proc)
                logfile = tempfile.NamedTemporaryFile(mode="w", prefix=f"HR_RUNNER_{proc.pid}_", suffix=".log")
                logger.info(f"PID {proc.pid}; check stdout at {logfile.name}")
                with logfile as f:
                    stdoutList = deque()
                    async for line in proc.stdout:
                        line = line.decode()
                        f.write(line)
                        f.flush()
                        line = line.strip()
                        logger.debug(line)
                        stdoutList.append(line)
                stdout = "\n".join(stdoutList)
                await proc.wait()
                proc_set.remove(proc)
                if proc.returncode != 0:
                    logger.error(f'{command_list} exited with returncode {proc.returncode}.')
                    if stdout:
                        logger.error(f'[stdout & stderr]\n{stdout}')
                    asyncio.create_task(message_queue.put(
                        (logging.ERROR, f"Command `{command_list}` exited with returncode `{proc.returncode}` "
                        f"and the following outputs: \n```\n{stdout}\n```"
                        ))
                    )
            else:
                logger.info(f'Running command: {command_list}')
                proc = await asyncio.create_subprocess_exec(command_list[0], *command_list[1:])
                proc_set.add(proc)
                stdout, stderr = await proc.communicate()
                proc_set.remove(proc)
                if proc.returncode != 0:
                    logger.error(f'{command_list} exited with returncode {proc.returncode}.')
                else:
                    logger.info(f'[{command_list} exited with returncode {proc.returncode}]')
            
            existingJobList = list(project.find_jobs(doc_filter=dict(comment=command, success=True)))
            assert len(existingJobList) == 1
            return existingJobList[0].id


async def exp_worker(worker_id, expConfigDf: pandas.DataFrame,
                     task_queue: asyncio.Queue, fail_queue: asyncio.Queue, message_queue: asyncio.Queue(),
                     is_running, csv_write_callback):
    logger = logging.getLogger(f"Worker-{worker_id}")
    logger.setLevel(logging.INFO)

    while True:
        # Get a "work item" out of the queue.
        task_id, task_data = await task_queue.get()
        logger.info(
            f"begin processing job {task_id} ====== \n{task_data}")
        is_running[worker_id] = True
        attack_session = task_data.AttackSession
        try:
            if expConfigDf.loc[task_id, "evasionJobID"] is None:
                # Run evasion, populate clean as well if missing
                logger.info(f"Begin to run evasion test for task {task_id}")
                jobID = await run_subprocess(attack_session, "evasion", task_data, logger, message_queue)
                expConfigDf.loc[task_id, "evasionJobID"] = jobID
                if expConfigDf.loc[task_id, "cleanJobID"] is None:
                    expConfigDf.loc[task_id, "cleanJobID"] = jobID
                csv_write_callback(expConfigDf)

            if expConfigDf.loc[task_id, "poisonJobID"] is None:
                if expConfigDf.loc[task_id, "cleanJobID"] is not None:
                    # Run poison only without testing clean
                    logger.info(f"Begin to run poison test for task {task_id}")
                    jobID = await run_subprocess(attack_session, "poison_only", task_data, logger, message_queue)
                    expConfigDf.loc[task_id, "poisonJobID"] = jobID
                else:
                    # Run poison and clean test
                    logger.info(f"Begin to run poison and clean test for task {task_id}")
                    jobID = await run_subprocess(attack_session, "poison", task_data, logger, message_queue)
                    expConfigDf.loc[task_id, "poisonJobID"] = jobID
                    expConfigDf.loc[task_id, "cleanJobID"] = jobID
                csv_write_callback(expConfigDf)

            if expConfigDf.loc[task_id, "cleanJobID"] is None:
                # Run clean only
                logger.info(f"Begin to run clean only test for task {task_id}")
                jobID = await run_subprocess(attack_session, "clean_only", task_data, logger, message_queue)
                expConfigDf.loc[task_id, "cleanJobID"] = jobID
                csv_write_callback(expConfigDf)

            if ("perturbJobID" in expConfigDf.columns 
                and expConfigDf.loc[task_id, "perturbJobID"] is None):
                logger.info(f"Begin to run perturbation task {task_id}")
                jobID = await run_subprocess(attack_session, "perturb", task_data, logger, message_queue)
                expConfigDf.loc[task_id, "perturbJobID"] = jobID
                csv_write_callback(expConfigDf)
                
        except Exception as e:
            strio = io.StringIO()
            traceback.print_exception(type(e), e, e.__traceback__, file=strio)
            logger.error(strio.getvalue())
            asyncio.create_task(message_queue.put((logging.ERROR, 
                f"Task {task_id} failed with following exception:\n```\n{strio.getvalue()}\n```")))
            asyncio.create_task(fail_queue.put((task_id, task_data)))
        # Notify the queue that the job has been processed.
        is_running[worker_id] = False
        task_queue.task_done()

async def scheduler_main(args, expConfigDf):
    logger = logging.getLogger("scheduler")
    logger.setLevel(logging.INFO)

    # Build task queues
    task_queue = asyncio.Queue()
    fail_queue = asyncio.Queue()
    message_queue = asyncio.Queue()
    is_running = [False] * args.parallel_num
    
    if not args.reverse_order:
        df_iter = expConfigDf.iterrows()
    else:
        df_iter = expConfigDf.iloc[::-1].iterrows()
    for task in df_iter:
        task_queue.put_nowait(task)
    try:
        workers_list = set()
        for worker_id in range(args.parallel_num):
            worker = asyncio.create_task(
                exp_worker(worker_id, expConfigDf, task_queue, fail_queue, message_queue,
                           is_running, lambda df: write_csv(df, args.result_path)))
            workers_list.add(worker)

        digest = create_digest(task_queue, fail_queue, message_queue, workers_list, expConfigDf)
        logger.info(f"Experiment begin: {digest}")
        digest_slack = create_digest(task_queue, fail_queue, message_queue, workers_list, expConfigDf, add_csv=False)
        asyncio.create_task(message_queue.put((logging.INFO, f"Experiment begin: {digest_slack}")))

        progress_monitor_task = asyncio.create_task(
            progress_monitor(task_queue, fail_queue, message_queue, workers_list, expConfigDf, is_running))
        slack_client_task = asyncio.create_task(slack_client(args, message_queue))
        queue_complete = asyncio.create_task(task_queue.join()) 
        while not queue_complete.done():
            await asyncio.wait([queue_complete, *workers_list], return_when=asyncio.FIRST_COMPLETED)
            if not queue_complete.done(): # Some worker has crashed
                new_worker_list = workers_list.copy()
                for worker in workers_list:
                    if worker.done():
                        try:
                            worker.result()
                        except Exception as e:
                            strio = io.StringIO()
                            traceback.print_exception(type(e), e, e.__traceback__, file=strio)
                            logger.error(strio.getvalue())
                            asyncio.create_task(message_queue.put((logging.ERROR, 
                                f"A worker failed with following exception:\n```\n{strio.getvalue()}\n```")))
                    new_worker_list.remove(worker)
                workers_list = new_worker_list
                if len(workers_list) == 0:
                    raise RuntimeError("All workers have crashed - please check the log for captured errors!")
            else: # Queue has finished
                digest = create_digest(task_queue, fail_queue, message_queue, workers_list, expConfigDf)
                digest_slack = digest
                #digest_slack = create_digest(task_queue, fail_queue, message_queue, workers_list, expConfigDf, add_csv=False)
                if args.auto_retry and fail_queue.qsize() > 0:
                    logger.info(f"Normal experiment running has finished; now retrying failed jobs one by one: {digest}")
                    asyncio.create_task(message_queue.put((
                        logging.INFO, f"Normal experiment running has finished; now retrying failed jobs one by one: {digest_slack}")))
                    asyncio.create_task(message_queue.put((
                        "direct-info", f"Normal experiment running has finished; now retrying failed jobs one by one: {digest_slack}")))
                    # Keep only 1 worker
                    new_worker_list = workers_list.copy()
                    for worker in list(workers_list)[1:]:
                        worker.cancel()
                        new_worker_list.remove(worker)
                    workers_list = new_worker_list
                    write_csv(expConfigDf, args.result_path)

                    while fail_queue.qsize() > 0:
                        task = fail_queue.get_nowait()
                        task_queue.put_nowait(task)
                    
                    # Recreate queue complete task
                    queue_complete = asyncio.create_task(task_queue.join())
                    args.auto_retry = False
                else:
                    logger.info(f"Experiment has finished: {digest}")
                    for worker in workers_list:
                        worker.cancel()
                    msg1_comp = message_queue.put((logging.INFO, f"Experiment has finished: {digest_slack}"))
                    msg2_comp = message_queue.put(("completed", f"{digest_slack}"))
                    await asyncio.gather(*workers_list, msg1_comp, msg2_comp, return_exceptions=True)
                    await message_queue.join()
                    progress_monitor_task.cancel()
                    slack_client_task.cancel()
                    write_csv(expConfigDf, args.result_path)
                
            
    except asyncio.CancelledError:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.critical("SIGINT received. Exiting...")
        for worker in workers_list:
            worker.cancel()
        queue_complete.cancel()
        progress_monitor_task.cancel()
        slack_client_task.cancel()
        for proc in proc_set:
            try:
                proc.send_signal(signal.SIGTERM)
                await proc.wait()
            except ProcessLookupError:
                pass
        write_csv(expConfigDf, args.result_path)


def main():
    '''
    Defense Experiments:
    1. Create a pandas dataframe with all variables and the corresponding perturbJobs ID. 
    2. Expand tables with models and their corresponding parameters to run. 
        - Evasion + Poison: run 1 evasion and 1 poison
        - Poison only: run 1 poison with also clean performance reported
    3. Sort the table using the priorities defined in the config file
    4. Save the table to a CSV file on disk as initial job tables
    5. Create a multiprocess pool to run the jobs based on the order as listed in the table
    6. Status monitoring and error handling
    '''
    args = parse_args()
    if args.vsdebug:
        import ptvsd
        print("Waiting for debugger attach. Press CTRL+C to skip.")
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()
    
    if args.dry_run:
        global dry_run_mode
        dry_run_mode = True
    if args.debug:
        global debug_mode
        debug_mode = True
    if args.get_status:
        global detect_mode
        detect_mode = True
    
    to_build_from_json = False

    if args.config.endswith(".json"):
        to_build_from_json = True
    elif args.config.endswith(".csv"):
        to_build_from_json = False
    else:
        raise ValueError(f"Input config file {args.config} is neither JSON or CSV format!")

    for pid in args.wait_pid:
        print(f"Waiting for process {pid} to finish...")
        os.system(f"tail -f /dev/null --pid {pid}")

    if to_build_from_json:
        expConfigDf = build_from_json(args.config)
    else:
        expConfigDf = pandas.read_csv(
            args.config, index_col=0, keep_default_na=False)
        expConfigDf.iloc[:, -3:] = expConfigDf.iloc[:, -3:].replace({"": None})
    result_folder = Path(args.config.replace(f"configs{os.sep}", f"results{os.sep}")).parent
    config_name = Path(args.config).stem
    args.result_path = args.result_path.format(result_folder=result_folder, config_name=config_name)
    if Path(args.result_path).exists():
        print(f"WARNING: A CSV state file already exist at {args.result_path}. Overwrite in 3s...", file=sys.stderr)
        for i in range(3):
            print(f"{3-i}s..")
            time.sleep(1)
        print()
    print(f"CSV will be saved to {args.result_path}")
    write_csv(expConfigDf, args.result_path)

    # Start async experiment job scheduler
    loop = asyncio.get_event_loop()
    scheduler_task = asyncio.ensure_future(scheduler_main(args, expConfigDf))
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, scheduler_task.cancel)
    try:
        loop.run_until_complete(scheduler_task)
    finally:
        loop.close()

if __name__ == "__main__":
    main()
