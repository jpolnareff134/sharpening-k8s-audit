# Sharpening Kubernetes Audit Logs with Context Awareness

## Description

This is the repository for the paper "Sharpening Kubernetes Audit Logs with Context Awareness". To streamline the 
Kubernetes audit logs, we aim to reconstruct contexts, i.e., identify actions performed by actors on the cluster plus 
all the supplementary events directly caused by these actions. 
Using inference rules and a Machine Learning model, correlated API calls are automatically identified, labeled, and 
consistently grouped together. We achieve contextual information while reducing the space on the disk by 99%.

## Installation

This project has been developed and tested using Python 3.10. To run it, and install the required Python libraries, you 
can follow these steps:

1. Clone the GitHub repository:
    ```bash
    git clone https://github.com/jpolnareff134/sharpening-k8s-audit.git
    ```

2. Navigate into the cloned repository:
    ```bash
    cd sharpening-k8s-audit
    ```

3. Create a new Python 3 virtual environment. Although not necessary, it is recommended to use a virtual environment to 
isolate the project dependencies:
    ```bash
    python3 -m venv env
    ```
   
4. Activate the virtual environment:
    ```bash
    source env/bin/activate
     ```

5. Install the dependencies using `requirements.txt` or `requirements-macos.txt` depending on your OS.  

    On a Linux-based distribution:
    ```bash
    pip3 install -r requirements.txt
    ```

    On a Mac system:
    ```bash
    pip3 install -r requirements-macos.txt
    ```
   
6. To speed up execution, if a GPU is available set CUDA-related environment variables with the following command:
    ```bash
    source .cudnn-env
    ```

## Usage

### Log parser

To assist users in manually creating datasets, the `log_parser.py` provides several functionalities that can be 
activated using the following flags:

```
    -f: The log input file. To use in conjunction with all the other flags.
    --labelling: Interactive labelling mode (default=False)
    --reduction: Reduction mode: remove unnecessary audit log lines (content + blacklisted) (default=False)
    --light-reduction: Light reduction mode: remove unnecessary audit log lines (content only) (default=False)
```

When the labelling mode is enabled this program uses the **Label proposer** capabilities to help users in manually 
labelling a dataset.

When reduction or light-reduction mode is chosen the script filters audit log based on rules specified in the **Notes**
section.

### Label proposer

To help users to manually assign labels this program provides some hints about labels to assign. The suggestions
depend on the resource, the subresource, the verb, the namespace and the status of the request contained in each audi 
log line. However, this script is not context-aware and is up to the user to understand where each context begins and 
ends to assign the correct label.

The following parameters can be specified when using `label_proposer.py`:

```
    -e, --encode: Encode to a label
    -f, --file: The log input file to label. To use in conjunction with --encode flag
    -d, --decode: Decode a label
    -l, --label: Label to decode. To use in conjunction with --decode flag
    --brute-force Print all the label space
```

### Model
The model is a deep, Recurrent Neural Network (RNN)-based model that takes as input a configurable amount of log lines 
in batches and outputs a label for each log line. 

The following parameters can be specified when using `model.py`:

```
    -f, --file: Path to the files, one or many. Globs are supported (e.g., /path/to/files/*.log)
    -m, --model: Path to the model file; if provided, will do inference instead of training
    -s, --stats-mode: Turns on statistics mode and accepts a comma-separated list of options.
                      save: saves the models generated during the process, else only statistics are saved.
                      kfolds: uses k-fold cross-validation instead of random splits.
    -l, --log-level: Log level (default='INFO')
    -G, --gpu: GPU to use, ignored if only one or no GPU is available (default=-1)
```

### Visualizer

Once the labels are predicted, the log lines can be visualized and queried. This can be done using the `visualizer.py`
script. This program shows a graph where the actions that happened in the system are shown. The displayed chart is 
interactive and its dots can be clicked. This causes a new window to appear showing the events in the context of the 
clicked action (dot).

The following parameters can be specified when using `visualizer.py`:

```
    -f, --file: The labelled log input file
    -p, --plot: Plot graph (default=False)
    -q, --query: Query to filter results
    -c, --csv: Dump all actions to a csv file (default=False)
    -k, --key: The key to use as label (default="label")
    -d, --dump: Dump the unclassified logs to the terminal (default=False)
```

For example, to display filter data based on a provided query, the following command can be used:

```bash
python3 visualizer.py -f path/to/labelled/data.log -p -q "username == user1 and exists(namespace)"
```

## Notes

### Logs filtered by Log parser

The Log parser filters out the following logs. The logs have been empirically determined after collecting logs from an 
empty Kubernetes cluster for three days.

1. #### Ignored namespaces and users

    The following namespaces are ignored:
    - `falco`
    - `kube-flannel`
  
    The following users/SAs are ignored:
    - `system:serviceaccount:kube-flannel:flannel`


2. #### Endpoints different from api/apis

    We will ignore endpoints which are not the regular API ones.

    - Verb: any
    - Users: any
    - Objects: any
    - Endpoints: any other than `/api` or `/apis`


3. #### API Server watching objects

   The API server monitors core components with a 10minute timeout. 
      - Verb: `watch`
     - Users: `system:apiserver`
     - Objects: `configmaps`, `clusterroles`, `namespaces`, `serviceaccounts`, `resourcequotas`, `clusterrolebindings`, 
`rolebindings`, `secrets`, `nodes`, `pods`, `roles`
     - Namespaces: none, apart from `configmaps` in `kube-system`. Additionally, if legacy service accounts are used, 
the API server also watches `kube-apiserver-legacy-service-account-token-tracking`, a CM in `kube-system`.


4. #### kube-controller-manager watching objects

    The `kube-system` namespace watches objects, similar to the API server. However, ConfigMaps are watched over 
non-namespaced objects and also `certificateigningrequests` are watched.

    - Verb: `watch`
    - Users: `system:kube-controller-manager`
    - Objects: same as the API server, plus `certificateigningrequests`
    - Namespaces: none


5. #### kube-controller-manager getting and creating tokens for GC and RQ controllers

    The `kube-controller-manager` gets every less than an hour the serviceaccounts 
`/api/v1/namespaces/kube-system/serviceaccounts/generic-garbage-collector` and 
`/api/v1/namespaces/kube-system/serviceaccounts/resourcequota-controller`. It then creates tokens for them, which 
expire in an hour.
    
    **5.1**
    - Verb: `get`
    - Users: `system:kube-controller-manager`, groups: `system:authenticated`
    - Objects: `/api/v1/namespaces/kube-system/serviceaccounts/generic-garbage-collector`
       
    **5.2**
    - Verb: `get`
    - Users: `system:kube-controller-manager`, groups: `system:authenticated`
    - Objects: `/api/v1/namespaces/kube-system/serviceaccounts/resourcequota-controller`

    **5.3**
    - Verb: `create`
    - Users: `system:kube-controller-manager`, groups: `system:authenticated`
    - Objects: `/api/v1/namespaces/kube-system/serviceaccounts/generic-garbage-collector/token`

    **5.4**
    - Verb: `create`
    - Users: `system:kube-controller-manager`, groups: `system:authenticated`
    - Objects: `/api/v1/namespaces/kube-system/serviceaccounts/resourcequota-controller/token`


6. #### Kube Scheduler watching objects

    **6.1 Pods, nodes, namespaces. Always with a 10-minute timeout.**
    - Verb: `watch`
    - Users: `system:kube-scheduler`
    - Objects: `pods`, `nodes`, `namespaces`
    - Namespaces: none

    **6.2 Extension-apiserver-authentication.**
    - Verb: `watch`
    - Users: `system:kube-scheduler`
    - Object: `configmaps/extension-apiserver-authentication`
    - Namespaces: `kube-system`


7. #### Nodes watching objects

    Each node in the cluster watches pods (non-namespaced), nodes
    (themselves), and some configmaps.

    **7.1 Pods**
    - Verb: `watch`
    - Users: `system:node:.*`, groups: `["system:nodes","system:authenticated"]`
    - Objects: `pods`
    - Namespaces: none

    **7.2 Nodes**
    - Verb: `watch`
    - Users: `system:node:.*`, groups: `["system:nodes","system:authenticated"]`
    - Objects: `nodes/{node-name}`
    - Namespaces: none

    **7.3  ConfigMaps**
    Every node hosting some Pod of any namespace will be in charge of watching the serviceaccounts of the Pods that are 
running. They of course depend on the components deployed in the cluster.
    - Verb: `watch`
    - Users: `system:node:.*`, groups: ["system:nodes","system:authenticated"]`
    - Objects: `configmaps/kube-system/configmaps/{serviceaccount}`
    - Namespace: `kube-system`


8. #### Nodes creating tokens for SAs in the kube-system namespac

    Since nodes watch the configmaps of the Pods they are running, as the tokens of their service accounts expire, they 
will recreate them.
    - Verb: `create`
    - Users: `system:node:.*`, groups: `["system:nodes","system:authenticated"]`
    - Objects: `/api/v1/namespaces/kube-system/serviceaccounts/{serviceaccount}/token`

9. #### Nodes getting their own status

    Nodes poll their own status every ten seconds.
    - Verb: `get`
    - Users: `system:node:.*`, groups: `["system:nodes","system:authenticated"]`
    - Objects: `/api/v1/nodes/{the same node}`


10. #### Nodes patching their status to update conditions

    Nodes update their status (e.g., MemoryPressure, DiskPressure, PIDPressure) by patching their own status every five 
minutes.
    - Verb: `patch`
    - Users: `system:node:.*`, groups:  `["system:nodes","system:authenticated"]`
    - Objects: `/api/v1/nodes/{the same node}`


11. #### CoreDNS watching namespaces

    - Verb: `watch`
    - Users: `system:serviceaccount:kube-system:coredns`
    - Objects: `namespaces`


12. #### Kube-proxy watching nodes

    - Verb: `watch`
    - Users: `system:serviceaccount:kube-system:kube-proxy`
    - Objects: `nodes`


## License

This software is licensed under the Apache 2.0 license. See [LICENSE](LICENSE) for more details.
