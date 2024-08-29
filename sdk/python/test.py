from kubernetes import client as k8s_client
from kubeflow.training import V1ReplicaSpec, V1TFJob, V1TFJobSpec, V1RunPolicy
from kubeflow.training import client as training_client
from kubernetes.client.models import V1ObjectMeta, V1PodSpec, V1PodTemplateSpec

# 基本配置
name = "mnist"
namespace = "kubeflow-user-example-com"
container_name = "tensorflow"
training_image = "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0"
model_dir = "/mnt/model"
data_dir = "/mnt/data"
local_output_dir = "/mnt/local_output"
script_dir = "/mnt/scripts"

# 创建 V1Volume 对象
def create_volume(name, host_path):
    return k8s_client.V1Volume(
        name=name,
        host_path=k8s_client.V1HostPathVolumeSource(path=host_path)
    )

# 创建 V1VolumeMount 对象
def create_volume_mount(name, mount_path):
    return k8s_client.V1VolumeMount(name=name, mount_path=mount_path)

# 配置容器
container = k8s_client.V1Container(
    name=container_name,
    image=training_image,
    command=[
        "python",
        script_dir + "/mnist_with_summaries.py",
        "--log_dir=" + model_dir + "/logs",
        "--data_dir=" + data_dir,
        "--model_dir=" + model_dir,
        "--learning_rate=0.01",
        "--batch_size=150"
    ],
    volume_mounts=[
        create_volume_mount("model-volume", model_dir),
        create_volume_mount("data-volume", data_dir),
        create_volume_mount("local-output-volume", local_output_dir),
        create_volume_mount("script-volume", script_dir)
    ]
)

# 公共的 PodSpec 配置
pod_spec = V1PodSpec(
    containers=[container],
    volumes=[
        create_volume("model-volume", "/path/to/local/model"),
        create_volume("data-volume", "/path/to/local/data"),
        create_volume("local-output-volume", ""),
        create_volume("script-volume", "/path/to/local/scripts")
    ]
)

# 创建 V1ReplicaSpec 对象
def create_replica_spec(replicas):
    return V1ReplicaSpec(
        replicas=replicas,
        restart_policy="Never",
        template=V1PodTemplateSpec(spec=pod_spec)
    )

# 创建 TFJob
tfjob = V1TFJob(
    api_version="kubeflow.org/v1",
    kind="TFJob",
    metadata=V1ObjectMeta(name=name, namespace=namespace),
    spec=V1TFJobSpec(
        run_policy=V1RunPolicy(clean_pod_policy="None"),
        tf_replica_specs={
            "Chief": create_replica_spec(replicas=1),
            "Worker": create_replica_spec(replicas=2),
            "PS": create_replica_spec(replicas=1)
        }
    )
)

# 提交 TFJob
kf_client = training_client.KubeflowOrgV1TFJobClient()
kf_client.create(tfjob, namespace=namespace)
