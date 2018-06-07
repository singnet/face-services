import yaml
from subprocess import Popen, PIPE


def _debug_snet_call(p, output, err):
    if p.returncode != 0:
        print("snet call failed with exit code {}".format(p.returncode))
        print("stdout", output)
        print("stderr", err)
        raise Exception("snet call failed")


def snet_setup(service_name, max_price=1000000):
    print("Get {} service details from SingularityNET".format(service_name))

    p = Popen("snet registry query {}".format(service_name).split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    _debug_snet_call(p, output, err)

    agent_address = yaml.load(output)['record']['agent']

    print("  - agent address is {}".format(agent_address))

    endpoint_cmd_str = "snet contract Agent --at {} endpoint".format(agent_address)
    p = Popen(endpoint_cmd_str.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    _debug_snet_call(p, output, err)
    endpoint = yaml.load(output)

    print("  - agent endpoint is {}".format(endpoint))

    print("Funding job on SingularityNET")
    job_cmd_str = "snet agent --at {} create-jobs --number 1 --max-price {} --funded --signed --no-confirm".format(agent_address, max_price)
    p = Popen(job_cmd_str.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    _debug_snet_call(p, output, err)
    jobs = yaml.load(output)['jobs']
    job = jobs[0]
    job_address = job['job_address']
    job_signature = job['job_signature']

    print("  - job funded and at address {}".format(job_address))

    return endpoint, job_address, job_signature