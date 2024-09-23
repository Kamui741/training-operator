'''
Author: ChZheng
Date: 2024-09-23 14:55:43
LastEditTime: 2024-09-23 14:55:43
LastEditors: ChZheng
Description:为了方便对程序进行调试，可以使用 Kubernetes 的 Python SDK (`kubernetes` 库) 实现一些常见的 Kubernetes 操作，这些操作类似于使用 `kubectl` 命令行工具进行的操作。以下是一些常见资源操作的示例代码，包括对 Pods、Deployments、Services、Jobs 等的操作，以及针对 TFJob、PyTorchJob 和 MPIJob 的专用操作。

### 1. **获取所有 Pods 的列表**

```python
from kubernetes import client, config

def list_pods(namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace)
    for pod in pods.items:
        print(f"Pod name: {pod.metadata.name}, Status: {pod.status.phase}")

# 示例调用
list_pods(namespace='my-namespace')
```

### 2. **获取 Pod 的详细信息**

```python
def get_pod_details(pod_name, namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
    print(f"Pod name: {pod.metadata.name}")
    print(f"Pod status: {pod.status.phase}")
    print(f"Pod IP: {pod.status.pod_ip}")
    print(f"Containers: {pod.spec.containers}")

# 示例调用
get_pod_details(pod_name='my-pod', namespace='my-namespace')
```

### 3. **删除指定 Pod**

```python
def delete_pod(pod_name, namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
    print(f"Pod {pod_name} deleted.")

# 示例调用
delete_pod(pod_name='my-pod', namespace='my-namespace')
```

### 4. **获取 Pod 的日志**

```python
def get_pod_logs(pod_name, namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    log = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
    print(log)

# 示例调用
get_pod_logs(pod_name='my-pod', namespace='my-namespace')
```

### 5. **获取所有 Deployments 的列表**

```python
def list_deployments(namespace='default'):
    config.load_kube_config()
    v1 = client.AppsV1Api()
    deployments = v1.list_namespaced_deployment(namespace)
    for deployment in deployments.items:
        print(f"Deployment name: {deployment.metadata.name}, Replicas: {deployment.status.replicas}")

# 示例调用
list_deployments(namespace='my-namespace')
```

### 6. **获取 Deployment 的详细信息**

```python
def get_deployment_details(deployment_name, namespace='default'):
    config.load_kube_config()
    v1 = client.AppsV1Api()
    deployment = v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
    print(f"Deployment name: {deployment.metadata.name}")
    print(f"Replicas: {deployment.status.replicas}")
    print(f"Available replicas: {deployment.status.available_replicas}")

# 示例调用
get_deployment_details(deployment_name='my-deployment', namespace='my-namespace')
```

### 7. **删除 Deployment**

```python
def delete_deployment(deployment_name, namespace='default'):
    config.load_kube_config()
    v1 = client.AppsV1Api()
    v1.delete_namespaced_deployment(name=deployment_name, namespace=namespace)
    print(f"Deployment {deployment_name} deleted.")

# 示例调用
delete_deployment(deployment_name='my-deployment', namespace='my-namespace')
```

### 8. **获取所有 Services 的列表**

```python
def list_services(namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    services = v1.list_namespaced_service(namespace)
    for service in services.items:
        print(f"Service name: {service.metadata.name}, Type: {service.spec.type}")

# 示例调用
list_services(namespace='my-namespace')
```

### 9. **获取 Service 的详细信息**

```python
def get_service_details(service_name, namespace='default'):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    service = v1.read_namespaced_service(name=service_name, namespace=namespace)
    print(f"Service name: {service.metadata.name}")
    print(f"Service type: {service.spec.type}")
    print(f"Service ports: {service.spec.ports}")

# 示例调用
get_service_details(service_name='my-service', namespace='my-namespace')
```

### 10. **获取所有 Jobs 的列表**

```python
def list_jobs(namespace='default'):
    config.load_kube_config()
    v1 = client.BatchV1Api()
    jobs = v1.list_namespaced_job(namespace)
    for job in jobs.items:
        print(f"Job name: {job.metadata.name}, Status: {job.status.succeeded}")

# 示例调用
list_jobs(namespace='my-namespace')
```

### 11. **获取 Job 的详细信息**

```python
def get_job_details(job_name, namespace='default'):
    config.load_kube_config()
    v1 = client.BatchV1Api()
    job = v1.read_namespaced_job(name=job_name, namespace=namespace)
    print(f"Job name: {job.metadata.name}")
    print(f"Status: {job.status.conditions}")
    print(f"Succeeded: {job.status.succeeded}")

# 示例调用
get_job_details(job_name='my-job', namespace='my-namespace')
```

### 12. **获取 TFJob、PyTorchJob 和 MPIJob 的详细信息**

```python
def get_tfjob_details(tfjob_name, namespace='default'):
    config.load_kube_config()
    v1 = client.CustomObjectsApi()
    tfjob = v1.get_namespaced_custom_object(
        group='kubeflow.org',
        version='v1',
        namespace=namespace,
        plural='tfjobs',
        name=tfjob_name
    )
    print(f"TFJob name: {tfjob['metadata']['name']}")
    print(f"Status: {tfjob['status']}")

def get_pytorchjob_details(pytorchjob_name, namespace='default'):
    config.load_kube_config()
    v1 = client.CustomObjectsApi()
    pytorchjob = v1.get_namespaced_custom_object(
        group='kubeflow.org',
        version='v1',
        namespace=namespace,
        plural='pytorchjobs',
        name=pytorchjob_name
    )
    print(f"PyTorchJob name: {pytorchjob['metadata']['name']}")
    print(f"Status: {pytorchjob['status']}")

def get_mpijob_details(mpijob_name, namespace='default'):
    config.load_kube_config()
    v1 = client.CustomObjectsApi()
    mpijob = v1.get_namespaced_custom_object(
        group='kubeflow.org',
        version='v1',
        namespace=namespace,
        plural='mpijobs',
        name=mpijob_name
    )
    print(f"MPIJob name: {mpijob['metadata']['name']}")
    print(f"Status: {mpijob['status']}")

# 示例调用
get_tfjob_details(tfjob_name='my-tfjob', namespace='my-namespace')
get_pytorchjob_details(pytorchjob_name='my-pytorchjob', namespace='my-namespace')
get_mpijob_details(mpijob_name='my-mpijob', namespace='my-namespace')
```

以上代码展示了如何使用 Kubernetes Python SDK 执行常见的 Kubernetes 操作，包括查看、操作和调试各种 Kubernetes 资源。你可以根据具体需要修改命名空间和资源名称，或将这些函数整合到你的调试工具中。
FilePath: /horovod/Users/apple/go/src/github.com/training-operator/sdkjob/tools.py
'''
要实现列出 Kubernetes 资源的 YAML 文件功能，你可以使用 Kubernetes Python SDK 中的 `CustomObjectsApi` 和其他 API 类来获取资源的详细信息，并将其格式化为 YAML 格式。以下是一个实现示例，包括对 Pods、Deployments、Services、Jobs、TFJobs、PyTorchJobs 和 MPIJobs 的资源进行 YAML 输出。

### 1. **安装所需库**

确保你已经安装了 `pyyaml` 库，用于将 Python 对象转换为 YAML 格式。你可以通过以下命令安装它：

```bash
pip install pyyaml
```

### 2. **实现获取资源的 YAML 文件功能**

```python
import yaml
from kubernetes import client, config

def get_yaml(resource, kind, namespace='default'):
    """获取资源的 YAML 文件"""
    config.load_kube_config()
    api_instance = None

    if kind == 'Pod':
        api_instance = client.CoreV1Api()
        resource = api_instance.read_namespaced_pod(name=resource, namespace=namespace)
    elif kind == 'Deployment':
        api_instance = client.AppsV1Api()
        resource = api_instance.read_namespaced_deployment(name=resource, namespace=namespace)
    elif kind == 'Service':
        api_instance = client.CoreV1Api()
        resource = api_instance.read_namespaced_service(name=resource, namespace=namespace)
    elif kind == 'Job':
        api_instance = client.BatchV1Api()
        resource = api_instance.read_namespaced_job(name=resource, namespace=namespace)
    elif kind == 'TFJob':
        api_instance = client.CustomObjectsApi()
        resource = api_instance.get_namespaced_custom_object(
            group='kubeflow.org',
            version='v1',
            namespace=namespace,
            plural='tfjobs',
            name=resource
        )
    elif kind == 'PyTorchJob':
        api_instance = client.CustomObjectsApi()
        resource = api_instance.get_namespaced_custom_object(
            group='kubeflow.org',
            version='v1',
            namespace=namespace,
            plural='pytorchjobs',
            name=resource
        )
    elif kind == 'MPIJob':
        api_instance = client.CustomObjectsApi()
        resource = api_instance.get_namespaced_custom_object(
            group='kubeflow.org',
            version='v1',
            namespace=namespace,
            plural='mpijobs',
            name=resource
        )
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    return yaml.dump(resource.to_dict())

# 示例调用
namespace = 'my-namespace'
print(get_yaml(resource='my-pod', kind='Pod', namespace=namespace))
print(get_yaml(resource='my-deployment', kind='Deployment', namespace=namespace))
print(get_yaml(resource='my-service', kind='Service', namespace=namespace))
print(get_yaml(resource='my-job', kind='Job', namespace=namespace))
print(get_yaml(resource='my-tfjob', kind='TFJob', namespace=namespace))
print(get_yaml(resource='my-pytorchjob', kind='PyTorchJob', namespace=namespace))
print(get_yaml(resource='my-mpijob', kind='MPIJob', namespace=namespace))
```

### 3. **详细说明**

1. **`get_yaml` 函数**：此函数接收资源名称、资源类型（如 Pod、Deployment、Service 等）和命名空间，然后使用相应的 API 获取资源的详细信息，并将其转换为 YAML 格式。

2. **资源类型**：
   - `Pod`、`Deployment`、`Service`、`Job` 使用 Kubernetes API 的标准客户端获取。
   - `TFJob`、`PyTorchJob`、`MPIJob` 使用 CustomObjectsApi 来处理 Kubeflow 的自定义资源。

3. **`yaml.dump`**：将 Python 对象转换为 YAML 格式的字符串。

### 4. **如何使用**

将上述代码保存为一个 Python 脚本，并根据实际情况修改资源名称和命名空间。运行脚本时，它将输出指定资源的 YAML 配置。

你可以将这些功能整合到调试工具中，以便在进行资源排查和调试时方便地获取资源的详细配置。