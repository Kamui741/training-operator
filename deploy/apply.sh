###
###
 # @Author: ChZheng
 # @Date: 2024-06-21 17:11:26
 # @LastEditTime: 2024-06-25 16:01:03
 # @LastEditors: ChZheng
 # @Description:
 # @FilePath: /training-operator/deploy/apply.sh
###

echo "开始部署..."
kubectl apply -f namespace.yaml



kubectl apply -f crd_tfjobs.yaml
kubectl apply -f crd_mxjobs.yaml
kubectl apply -f crd_pytorchjobs.yaml
kubectl apply -f crd_xgboostjobs.yaml
kubectl apply -f crd_mpijobs.yaml
kubectl apply -f crd_paddlejobs.yaml

kubectl apply -f service-account.yaml #顺序
kubectl apply -f cluster-role.yaml
kubectl apply -f cluster-role-binding.yaml
kubectl apply -f init.yaml
kubectl apply -f webhook.yaml
kubectl apply -f service.yaml

kubectl apply -f deployment.yaml
echo "部署完成。"