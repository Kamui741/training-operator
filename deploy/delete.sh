###
 # @Author: ChZheng
 # @Date: 2024-05-23
 # @Description: 删除 Kubeflow Training Operator 相关资源
###

echo "开始删除..."

kubectl delete -f deployment.yaml
kubectl delete -f service.yaml
kubectl delete -f webhook.yaml
kubectl delete -f crd_paddlejobs.yaml
kubectl delete -f crd_mpijobs.yaml
kubectl delete -f crd_xgboostjobs.yaml
kubectl delete -f crd_pytorchjobs.yaml
kubectl delete -f crd_mxjobs.yaml
kubectl delete -f crd_tfjobs.yaml
kubectl delete -f init.yaml
kubectl delete -f cluster-role-binding.yaml
kubectl delete -f role.yaml
kubectl delete -f service-account.yaml
kubectl delete -f namespace.yaml

echo "删除完成。"
