# trainer

* BasicTrainer
    * NormalTrainer
    * IterationTrainer
    
## BasicTrainer
the base abstract class  
基础模型的训练类。

### NormalTrainer
trainer that train model by calling `model.train` once.  
调用`model.train`一次的训练器

### IterationTrainer
trainer that train model with iteration  
调用`model.train_batch`来迭代多次训练模型

