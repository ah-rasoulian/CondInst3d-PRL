import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from condinst3d.arch.condinst3d_prl import CondInst3dPRL
from condinst3d.arch.backbone.unetr_specific_mod import UNetRSpecificMod
import torch

@hydra.main(config_path="../condinst3d/conf", config_name="condinst3d-prl", version_base=None)
def main(cfg: DictConfig):
    model = CondInst3dPRL(cfg.model)
    # trainer = Trainer(**cfg.trainer)
    # dm = instantiate(cfg.datamodule)
    # trainer.fit(model, datamodule=dm)
    x = torch.randn(4, 4, 96, 96, 64, device='cuda')
    t = [
        {'boxes': torch.tensor([[24, 24, 24, 32, 32, 32]], dtype=torch.float32, device='cuda'), 'classes': torch.tensor([0], device='cuda'), 'onehot': torch.randint(0, 2, (1, 1, 96, 96, 64), device='cuda', dtype=torch.bool)},
        {'boxes': torch.tensor([[24, 24, 24, 32, 32, 32]], dtype=torch.float32, device='cuda'), 'classes': torch.tensor([0], device='cuda'), 'onehot': torch.randint(0, 2, (1, 1, 96, 96, 64), device='cuda', dtype=torch.bool)},
        {'boxes': torch.tensor([[24, 24, 24, 32, 32, 32]], dtype=torch.float32, device='cuda'), 'classes': torch.tensor([0], device='cuda'), 'onehot': torch.randint(0, 2, (1, 1, 96, 96, 64), device='cuda', dtype=torch.bool)},
        {'boxes': torch.tensor([[24, 24, 24, 32, 32, 32]], dtype=torch.float32, device='cuda'), 'classes': torch.tensor([0], device='cuda'), 'onehot': torch.randint(0, 2, (1, 1, 96, 96, 64), device='cuda', dtype=torch.bool)},
    ]
    model.cuda()
    y = model.predict_step({'inputs': x, 'targets': t}, 0)
    print(y)

if __name__ == "__main__":
    main()