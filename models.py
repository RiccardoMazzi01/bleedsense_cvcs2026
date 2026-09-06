import random

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


class MixStyle(nn.Module):
    """Domain Generalization with MixStyle (Zhou et al., ICLR 2021).

    Mescola media/deviazione standard per canale delle feature map tra coppie di
    campioni nello stesso batch: ogni campione viene normalizzato con le proprie
    statistiche (come una instance norm) e poi ri-scalato con statistiche miste
    (combinazione convessa con quelle di un altro campione del batch, peso ~Beta(alpha,alpha)).
    A differenza di Reinhard/FDA (Fase 3, sempre attivi sui pixel), qui l'intervento e'
    a livello di feature dentro l'encoder, probabilistico (attivo solo in training, con
    probabilita' p per batch) e non richiede immagini del dominio target.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or x.size(0) < 2 or random.random() > self.p:
            return x

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig

        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((x.size(0), 1, 1, 1)).to(x.device)
        perm = torch.randperm(x.size(0), device=x.device)

        mu_mix = mu * lmda + mu[perm] * (1 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1 - lmda)

        return x_normed * sig_mix + mu_mix


def attach_mixstyle(model, layers=("layer1", "layer2"), p=0.5, alpha=0.1):
    """Aggancia MixStyle come forward hook sui blocchi indicati dell'encoder ResNet.

    Ogni MixStyle e' registrato come sottomodulo di `model` (non solo catturato in una
    closure), cosi' model.train()/model.eval() ne propaga correttamente lo stato.
    """
    for name in layers:
        block = getattr(model.encoder, name, None)
        if block is None:
            raise ValueError(f"Layer encoder '{name}' non trovato: MixStyle richiede un encoder "
                              f"stile ResNet (layer1..layer4)")
        mixstyle = MixStyle(p=p, alpha=alpha)
        setattr(model, f"_mixstyle_{name}", mixstyle)
        block.register_forward_hook(lambda m, inp, out, ms=mixstyle: ms(out))
    return model


def get_model(architecture, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1,
              mixstyle_layers=None, mixstyle_p=0.5, mixstyle_alpha=0.1):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Architettura sconosciuta: {architecture}. Disponibili: {list(ARCHITECTURES)}")

    model_cls = ARCHITECTURES[architecture]
    model = model_cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # restiamo in logit space, la sigmoid e' gestita da loss/metriche
    )

    if mixstyle_layers:
        attach_mixstyle(model, layers=mixstyle_layers, p=mixstyle_p, alpha=mixstyle_alpha)

    return model
