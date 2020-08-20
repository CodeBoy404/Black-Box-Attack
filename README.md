# Black-Box-Attack
黑盒对抗攻击项目 --AHBA算法
=======

model是模型的网络结构

pretrained_model是预训练的模型

dataset.py：导入tinyimagenet数据集

AHBA_attack.py：调用具体攻击方式，并实现边界搜索和二分搜索

AHBA_targeted_attack.py：有目标攻击

AHBA_untargeted_attack_tiny.py：无目标攻击

better_dnn.py：改进的dnn

cifar10_test.py：在cifar10中进行无目标攻击测试

cifar100_test.py：在cifar100中进行无目标攻击测试

