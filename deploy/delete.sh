###
 # @Author: ChZheng
 # @Date: 2024-05-14 19:33:41
 # @LastEditTime: 2024-05-14 19:33:43
 # @LastEditors: ChZheng
 # @Description:
 # @FilePath: /training-operator/deploy/delete.sh
###
echo "开始删除..."
kubectl delete -f deployment.yaml
kubectl delete secret training-operator-webhook-cert -n kubeflow
kubectl delete -f webhook.yaml
kubectl delete -f kubeflow.org_mpijobs.yaml
kubectl delete -f rbac.yaml
kubectl delete -f namespace.yaml
echo "删除完成。"