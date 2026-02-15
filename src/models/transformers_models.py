from transformers import ViTConfig, ViTForImageClassification

class ViTWrapper(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.model = vit_model

    def forward(self, x):
        return self.model(x).logits   # retourne directement le Tensor
    

def LoadViT():
    config = ViTConfig(hidden_size=192,
                            num_hidden_layers=9,
                            num_attention_heads=3,
                            intermediate_size=768, # intermediate_size = hidden_size * 4
                            hidden_dropout_prob = 0.1,
                            attention_dropout = 0.1,
                            image_size=32,      # CIFAR100
                            num_labels=100,
                            patch_size=4,
                            num_channels=3,
                            qkv_bias=True,
                            classifier="token"
                            )
    model = ViTForImageClassification(config)
    return ViTWrapper(model)


